from __future__ import annotations

import base64
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from PIL import Image

from .config import Settings
from .models import AdInput, ClipSignals, ImageSignals, LLMReview, LocalTextSignals


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = _JSON_BLOCK.search(text)
    candidate = match.group(1).strip() if match else text
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model did not return a JSON object.")
        result = json.loads(candidate[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("The model response was not a JSON object.")
    return result


def _compact_ad(ad: AdInput) -> str:
    return ad.full_text[:10_000]


def _image_data_url(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    rgb.thumbnail((768, 768))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=76, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class OpenRouterJury:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def available(self) -> bool:
        return bool(self.settings.openrouter_api_key)

    def _request(
        self,
        models: list[str],
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.15,
        max_tokens: int = 900,
    ) -> tuple[dict[str, Any], str, list[str]]:
        errors: list[str] = []
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.app_url,
            "X-Title": self.settings.app_name,
        }
        for model in models:
            try:
                response = httpx.post(
                    self.settings.openrouter_url,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=self.settings.request_timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                return _extract_json(str(content)), model, errors
            except Exception as exc:  # Providers can fail independently; try the next model.
                errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:180]}")
        return {}, "", errors

    def light_review(self, ad: AdInput) -> LLMReview:
        if not self.available:
            return LLMReview(enabled=False)

        visual_messages = [
            {
                "role": "system",
                "content": (
                    "You are the fast visual-query specialist for a classifieds quality system. "
                    "Extract only concrete, visible claims. Translate Persian when necessary. "
                    "Return JSON only with: visual_caption (one concise English sentence optimized "
                    "for CLIP image matching), expected_objects (array), non_visual_claims (array), "
                    "and ambiguity_notes (array). Do not invent details."
                ),
            },
            {"role": "user", "content": _compact_ad(ad)},
        ]
        copy_messages = [
            {
                "role": "system",
                "content": (
                    "You are the fast copy and safety specialist for a person-to-person marketplace. "
                    "Assess the supplied ad text, not the photos. Return JSON only with numeric "
                    "copy_quality_score and trust_score (0..100), plus arrays missing_details, "
                    "risk_flags, and rewrite_tips. Penalize vague copy, unverifiable superlatives, "
                    "off-platform contact/payment, manipulation, and contradictions."
                ),
            },
            {"role": "user", "content": _compact_ad(ad)},
        ]

        visual_models = self.settings.light_models
        copy_models = self.settings.light_models[1:] + self.settings.light_models[:1]
        with ThreadPoolExecutor(max_workers=2) as pool:
            visual_future = pool.submit(self._request, visual_models, visual_messages)
            copy_future = pool.submit(self._request, copy_models, copy_messages)
            visual, visual_model, visual_errors = visual_future.result()
            copy, copy_model, copy_errors = copy_future.result()

        models_used = [model for model in (visual_model, copy_model) if model]
        return LLMReview(
            enabled=bool(models_used),
            models_used=models_used,
            visual_brief=visual,
            copy_review=copy,
            errors=visual_errors + copy_errors,
        )

    def heavy_review(
        self,
        ad: AdInput,
        clip: ClipSignals,
        image: ImageSignals,
        local_text: LocalTextSignals,
        review: LLMReview,
    ) -> LLMReview:
        if not self.available:
            return review

        evidence = {
            "clip_text_image_score": clip.text_image_score,
            "clip_image_cohesion_score": clip.image_cohesion_score,
            "per_image_relevance": clip.per_image_relevance,
            "possible_outlier_images_1_based": [index + 1 for index in clip.outlier_indices],
            "image_quality_score": image.quality_score,
            "per_image_quality": image.per_image_quality,
            "local_completeness_score": local_text.completeness_score,
            "local_trust_score": local_text.trust_score,
            "fast_visual_specialist": review.visual_brief,
            "fast_copy_specialist": review.copy_review,
        }
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "AD\n"
                    + _compact_ad(ad)
                    + "\n\nMACHINE EVIDENCE\n"
                    + json.dumps(evidence, ensure_ascii=False)
                ),
            }
        ]
        for index, source in enumerate(ad.images[:4]):
            content.append({"type": "text", "text": f"Uploaded image {index + 1}:"})
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(source)}})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the senior multimodal evidence judge for a Divar-style classifieds ad. "
                    "Reconcile the ad, visible photos, CLIP signals, deterministic checks, and fast "
                    "specialists. CLIP is evidence, not truth: correct it only when the visible images "
                    "clearly justify doing so. Look for item identity, photo consistency, disclosed "
                    "condition, contradictions, buyer usefulness, persuasion without hype, and fraud "
                    "risk. Return JSON only with holistic_score (0..100), semantic_integrity_score "
                    "(0..100), confidence (0..100), strengths (array), risks (array), recommendations "
                    "(array), and short_rationale. Never reward unsupported claims."
                ),
            },
            {"role": "user", "content": content},
        ]
        judge, model, errors = self._request(
            self.settings.heavy_models,
            messages,
            temperature=0.1,
            max_tokens=1200,
        )
        if model:
            review.models_used.append(model)
            review.judge_review = judge
            review.enabled = True
        review.errors.extend(errors)
        return review

