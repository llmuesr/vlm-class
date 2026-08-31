from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .models import AdInput, ClipSignals


def _clip_text_score(cosine: np.ndarray) -> np.ndarray:
    """
    Convert CLIP cosine similarity into a practical 0..100 score.

    These thresholds are marketplace-oriented heuristics, not
    probabilities and not calibrated fraud predictions.
    """

    values = np.asarray(cosine, dtype=np.float32)

    return np.clip(
        (values - 0.12) / 0.23 * 100.0,
        0.0,
        100.0,
    )


def _clip_image_score(cosine: float) -> float:
    """
    Convert image-to-image cosine similarity into a 0..100 score.
    """

    value = float(cosine)

    if not np.isfinite(value):
        return 0.0

    return float(
        np.clip(
            (value - 0.50) / 0.38 * 100.0,
            0.0,
            100.0,
        )
    )


def _safe_l2_normalize(
    values: Any,
    axis: int = -1,
    epsilon: float = 1e-12,
) -> Any:
    """
    Normalize vectors while preventing division by zero.
    """

    norms = np.linalg.norm(
        values,
        axis=axis,
        keepdims=True,
    )

    norms = np.maximum(norms, epsilon)

    return values / norms


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback

    if not np.isfinite(result):
        return fallback

    return result


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _clean_caption(text: str, max_chars: int = 600) -> str:
    """
    Keep the caption compact because CLIP text tokenizers have
    a relatively small context window.
    """

    cleaned = _safe_text(text)

    if not cleaned:
        return "a clear marketplace product listing photo"

    return cleaned[:max_chars]


def _build_visual_caption(ad: AdInput) -> str:
    """
    Build a compact English visual query.

    CLIP checkpoints such as OpenAI CLIP generally work better with
    short visual descriptions than with a long raw Persian listing.
    """

    category = _safe_text(ad.category)
    title = _safe_text(ad.title)
    condition = _safe_text(ad.condition)

    visual_parts = [
        "a clear marketplace listing photo",
        f"showing the item described as {title}" if title else "",
        f"in the category {category}" if category else "",
        (
            f"with visible condition described as {condition}"
            if condition and condition != "نامشخص"
            else ""
        ),
        "showing the actual item for sale",
    ]

    caption = ". ".join(
        part
        for part in visual_parts
        if part
    )

    return _clean_caption(caption)


def _build_ad_context_caption(ad: AdInput) -> str:
    """
    Optional structured context for diagnostics and future prompt use.

    This is intentionally not used as the primary visual query because
    details such as city, price, ownership, and payment method are not
    normally visible in an image.
    """

    fields = [
        f"title={_safe_text(ad.title)}",
        f"category={_safe_text(ad.category)}",
        f"condition={_safe_text(ad.condition)}",
        f"usage={_safe_text(ad.usage_duration)}",
        f"year={ad.manufacture_year or ''}",
        f"defects={_safe_text(ad.defects)}",
        f"accessories={_safe_text(ad.included_items)}",
    ]

    return " | ".join(
        item
        for item in fields
        if item.split("=", 1)[1].strip()
    )


def _prepare_image(image: Image.Image) -> Image.Image:
    """
    Convert an uploaded image into a consistent RGB PIL image.
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            f"Expected PIL.Image.Image, got {type(image)!r}"
        )

    image.load()

    return image.convert("RGB")


def _extract_image_features(
    model: Any,
    inputs: dict[str, Any],
) -> Any:
    """
    Extract image features across compatible Transformers versions.
    """

    pixel_values = inputs["pixel_values"]

    try:
        return model.get_image_features(
            pixel_values=pixel_values,
        )
    except TypeError:
        return model.get_image_features(
            pixel_values,
        )


def _extract_text_features(
    model: Any,
    inputs: dict[str, Any],
) -> Any:
    """
    Extract text features across compatible Transformers versions.
    """

    text_args: dict[str, Any] = {
        "input_ids": inputs["input_ids"],
    }

    if "attention_mask" in inputs:
        text_args["attention_mask"] = inputs["attention_mask"]

    try:
        return model.get_text_features(**text_args)
    except TypeError:
        return model.get_text_features(
            inputs["input_ids"],
            inputs.get("attention_mask"),
        )


def _find_outliers(
    relevance: np.ndarray,
    image_embeddings: np.ndarray,
) -> list[int]:
    """
    Detect images that are weakly related to the text and/or
    inconsistent with the rest of the image set.
    """

    image_count = len(relevance)

    if image_count == 0:
        return []

    if image_count == 1:
        return [0] if float(relevance[0]) < 30.0 else []

    similarity_matrix = image_embeddings @ image_embeddings.T
    cohesion_values: list[float] = []

    for index in range(image_count):
        peers = [
            float(similarity_matrix[index, other])
            for other in range(image_count)
            if other != index
        ]

        if peers:
            cohesion_values.append(
                _clip_image_score(
                    float(np.mean(peers))
                )
            )
        else:
            cohesion_values.append(0.0)

    relevance_median = float(np.median(relevance))
    cohesion_median = float(np.median(cohesion_values))

    relevance_floor = max(
        30.0,
        relevance_median - 24.0,
    )

    cohesion_floor = max(
        25.0,
        cohesion_median - 30.0,
    )

    outliers: list[int] = []

    for index, relevance_score in enumerate(relevance):
        weak_relevance = (
            float(relevance_score)
            < relevance_floor
        )

        weak_cohesion = (
            cohesion_values[index]
            < cohesion_floor
        )

        # A very weak relevance score is sufficient by itself.
        # A moderate relevance score requires cohesion evidence too.
        if weak_relevance or (
            weak_cohesion
            and float(relevance_score) < 58.0
        ):
            outliers.append(index)

    return outliers


def _find_near_duplicates(
    image_embeddings: np.ndarray,
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Find highly similar image pairs and return pair quality scores.
    """

    image_count = len(image_embeddings)

    if image_count < 2:
        return [], []

    similarity_matrix = image_embeddings @ image_embeddings.T

    near_duplicates: list[tuple[int, int]] = []
    pair_scores: list[float] = []

    for left in range(image_count):
        for right in range(left + 1, image_count):
            cosine = _safe_float(
                similarity_matrix[left, right]
            )

            pair_scores.append(
                _clip_image_score(cosine)
            )

            if cosine >= 0.985:
                near_duplicates.append((left, right))

    return near_duplicates, pair_scores


def _calculate_cohesion(
    image_embeddings: np.ndarray,
    pair_scores: list[float],
) -> float:
    """
    Calculate image-set cohesion.

    The lower percentile prevents one unrelated image from being
    hidden by a high average. Very small image sets receive a
    conservative fallback.
    """

    image_count = len(image_embeddings)

    if image_count <= 1:
        return 62.0

    if not pair_scores:
        return 0.0

    average = float(np.mean(pair_scores))
    lower_percentile = float(
        np.percentile(pair_scores, 20)
    )

    cohesion = (
        0.65 * average
        + 0.35 * lower_percentile
    )

    return max(
        0.0,
        min(100.0, cohesion),
    )


@dataclass
class ClipAnalyzer:
    model_id: str = "openai/clip-vit-base-patch32"

    # Heuristic controls
    duplicate_cosine_threshold: float = 0.985
    max_caption_chars: int = 600

    def __post_init__(self) -> None:
        """
        Load the model lazily when ClipAnalyzer is instantiated.
        """

        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.processor = CLIPProcessor.from_pretrained(
            self.model_id
        )

        self.model = CLIPModel.from_pretrained(
            self.model_id
        ).to(self.device)

        self.model.eval()

    def analyze(
        self,
        images: list[Image.Image],
        visual_caption: str,
        ad: AdInput | None = None,
    ) -> ClipSignals:
        """
        Analyze text-image relevance, image cohesion,
        outliers, and near-duplicate images.

        `ad` is optional for backward compatibility. If supplied,
        it is used to build a stronger visual caption when the
        caller does not provide one.
        """

        if not images:
            fallback_caption = (
                visual_caption.strip()
                or (
                    _build_visual_caption(ad)
                    if ad is not None
                    else "a clear marketplace product listing photo"
                )
            )

            return ClipSignals(
                visual_caption_used=_clean_caption(
                    fallback_caption,
                    self.max_caption_chars,
                )
            )

        valid_images: list[Image.Image] = []
        skipped_images: list[int] = []

        for index, image in enumerate(images):
            try:
                valid_images.append(
                    _prepare_image(image)
                )
            except (
                AttributeError,
                OSError,
                TypeError,
                ValueError,
            ):
                skipped_images.append(index)

        if not valid_images:
            return ClipSignals(
                visual_caption_used=_clean_caption(
                    visual_caption
                    or (
                        _build_visual_caption(ad)
                        if ad is not None
                        else "a clear marketplace product listing photo"
                    ),
                    self.max_caption_chars,
                ),
                outlier_indices=list(
                    range(len(images))
                ),
            )

        if visual_caption.strip():
            text = _clean_caption(
                visual_caption,
                self.max_caption_chars,
            )
        elif ad is not None:
            text = _build_visual_caption(ad)
        else:
            text = "a clear marketplace product listing photo"

        inputs = self.processor(
            text=[text],
            images=valid_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }

        with self._torch.inference_mode():
            image_features = _extract_image_features(
                self.model,
                inputs,
            )

            text_features = _extract_text_features(
                self.model,
                inputs,
            )

        image_features = image_features.float()
        text_features = text_features.float()

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)

        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)

        image_np = (
            image_features
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        text_np = (
            text_features
            .detach()
            .cpu()
            .numpy()[0]
            .astype(np.float32)
        )

        image_np = _safe_l2_normalize(
            image_np
        )

        text_np = _safe_l2_normalize(
            text_np
        )

        raw_relevance = image_np @ text_np
        raw_relevance = np.nan_to_num(
            raw_relevance,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        relevance = _clip_text_score(
            raw_relevance
        )

        near_duplicates, pair_scores = (
            _find_near_duplicates(image_np)
        )

        # Respect the configured duplicate threshold even if the
        # helper's default threshold is different.
        if (
            self.duplicate_cosine_threshold
            != 0.985
        ):
            similarity_matrix = image_np @ image_np.T
            near_duplicates = []

            for left in range(len(valid_images)):
                for right in range(left + 1, len(valid_images)):
                    cosine = _safe_float(
                        similarity_matrix[left, right]
                    )

                    if (
                        cosine
                        >= self.duplicate_cosine_threshold
                    ):
                        near_duplicates.append(
                            (left, right)
                        )

        cohesion = _calculate_cohesion(
            image_embeddings=image_np,
            pair_scores=pair_scores,
        )

        outliers = _find_outliers(
            relevance=relevance,
            image_embeddings=image_np,
        )

        # If invalid files were skipped, restore their original
        # indices as outliers. Valid image indexes are mapped back
        # to the original upload indexes.
        mapped_outliers = [
            skipped_images[index]
            for index in range(len(skipped_images))
        ]

        valid_outliers = [
            index
            for index in outliers
            if 0 <= index < len(valid_images)
        ]

        for valid_index in valid_outliers:
            original_index = valid_index

            for skipped_index in skipped_images:
                if skipped_index <= original_index:
                    original_index += 1

            mapped_outliers.append(original_index)

        mapped_outliers = sorted(
            set(mapped_outliers)
        )

        if len(valid_images) == 1:
            relevance_score = float(relevance[0])
        else:
            relevance_score = (
                0.72 * float(np.mean(relevance))
                + 0.28 * float(np.min(relevance))
            )

        relevance_score = max(
            0.0,
            min(100.0, relevance_score),
        )

        mapped_duplicate_pairs: list[tuple[int, int]] = []

        for left, right in near_duplicates:
            original_left = left
            original_right = right

            for skipped_index in skipped_images:
                if skipped_index <= original_left:
                    original_left += 1

                if skipped_index <= original_right:
                    original_right += 1

            mapped_duplicate_pairs.append(
                (
                    original_left,
                    original_right,
                )
            )

        return ClipSignals(
            text_image_score=round(
                relevance_score,
                1,
            ),
            image_cohesion_score=round(
                max(0.0, min(100.0, cohesion)),
                1,
            ),
            per_image_relevance=[
                round(float(score), 1)
                for score in relevance
            ],
            outlier_indices=mapped_outliers,
            near_duplicate_pairs=mapped_duplicate_pairs,
            raw_text_image_cosines=[
                round(float(score), 4)
                for score in raw_relevance
            ],
            visual_caption_used=text,
        )