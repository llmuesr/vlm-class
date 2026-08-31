from __future__ import annotations

from typing import Any

from .models import (
    AdInput,
    AnalysisResult,
    ClipSignals,
    ImageSignals,
    LLMReview,
    LocalTextSignals,
    ScoreComponent,
)


def _number(
    data: dict[str, Any],
    key: str,
    fallback: float,
) -> float:
    """
    Read a numeric value from an LLM response and clamp it to 0..100.
    """

    if not isinstance(data, dict):
        return max(0.0, min(100.0, fallback))

    try:
        value = float(data.get(key, fallback))
    except (TypeError, ValueError):
        value = fallback

    return max(0.0, min(100.0, value))


def _first_number(
    data: dict[str, Any],
    keys: tuple[str, ...],
    fallback: float,
) -> float:
    """
    Read the first available numeric key from an LLM response.
    """

    if not isinstance(data, dict):
        return max(0.0, min(100.0, fallback))

    for key in keys:
        if key in data:
            return _number(data, key, fallback)

    return max(0.0, min(100.0, fallback))


def _strings(value: Any) -> list[str]:
    """
    Safely normalize a list returned by an LLM.
    """

    if not isinstance(value, list):
        return []

    output: list[str] = []

    for item in value:
        text = str(item).strip()

        if text:
            output.append(text)

    return output


def _dedupe(items: list[str]) -> list[str]:
    """
    Remove duplicate messages while preserving their original order.
    """

    seen: set[str] = set()
    output: list[str] = []

    for item in items:
        normalized = item.strip().casefold()

        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(item.strip())

    return output


def _average(values: list[float], fallback: float = 0.0) -> float:
    if not values:
        return fallback

    return sum(values) / len(values)


def _clip_relevance_score(clip: ClipSignals) -> float:
    """
    Score semantic alignment between the listing and its images.

    The aggregate CLIP score is preferred. Per-image relevance is used
    as a fallback when the aggregate value is unavailable.
    """

    aggregate = float(clip.text_image_score)

    if aggregate > 0:
        return max(0.0, min(100.0, aggregate))

    return max(
        0.0,
        min(
            100.0,
            _average(clip.per_image_relevance),
        ),
    )


def _cohesion_score(clip: ClipSignals) -> float:
    return max(
        0.0,
        min(100.0, float(clip.image_cohesion_score)),
    )


def _quality_score(image: ImageSignals) -> float:
    aggregate = float(image.quality_score)

    if aggregate > 0:
        return max(0.0, min(100.0, aggregate))

    return max(
        0.0,
        min(100.0, _average(image.per_image_quality)),
    )


def _completeness_score(local_text: LocalTextSignals) -> float:
    """
    Seller-information completeness.

    This includes title, description, category, location, price,
    product condition, usage history, defects, accessories,
    and transaction details as calculated by text_checks.py.
    """

    return max(
        0.0,
        min(
            100.0,
            float(local_text.completeness_score),
        ),
    )


def _transparency_score(local_text: LocalTextSignals) -> float:
    """
    Explicit transparency about defects, accessories, sale reason,
    inspection, receipt, and warranty.
    """

    value = float(local_text.transparency_score)

    if value <= 0:
        value = float(local_text.trust_score)

    return max(0.0, min(100.0, value))


def _local_trust_score(local_text: LocalTextSignals) -> float:
    return max(
        0.0,
        min(100.0, float(local_text.trust_score)),
    )


def _transaction_score(
    ad: AdInput,
    local_text: LocalTextSignals,
) -> float:
    """
    Calculate transaction trust from seller-provided details.

    This is not a fraud verdict. It only measures how clearly
    the transaction conditions are described.
    """

    score = 35.0

    if ad.transaction_method:
        score += 15.0

    if ad.delivery_options:
        score += 10.0

    if ad.viewing_available:
        score += 15.0

    if ad.original_receipt_available:
        score += 8.0

    if ad.warranty_available:
        score += 8.0

    if ad.city.strip():
        score += 4.0

    if ad.price.strip():
        score += 5.0

    if local_text.has_contact_info:
        score += 2.0

    if local_text.detected_sensitive_phrases:
        score -= min(
            30.0,
            len(local_text.detected_sensitive_phrases) * 8.0,
        )

    return max(0.0, min(100.0, score))


def _seller_context_score(
    ad: AdInput,
    local_text: LocalTextSignals,
) -> float:
    """
    Combine structured seller information into one score.

    This score rewards explicit information but does not assume
    that claims such as 'first owner' or 'warranty' are true.
    They are only treated as declared information.
    """

    fields = [
        ad.condition != "نامشخص",
        bool(ad.usage_duration.strip()),
        ad.manufacture_year is not None,
        ad.ownership_status != "نامشخص",
        ad.seller_type.strip() != "",
        bool(ad.defects.strip()),
        bool(ad.included_items.strip()),
        bool(ad.reason_for_selling.strip()),
        bool(ad.transaction_method),
        bool(ad.delivery_options),
        ad.viewing_available,
        ad.original_receipt_available,
        ad.warranty_available,
    ]

    declared_information_score = (
        sum(fields) / len(fields) * 100.0
    )

    completeness_score = _completeness_score(local_text)
    transparency_score = _transparency_score(local_text)

    score = (
        declared_information_score * 0.40
        + completeness_score * 0.30
        + transparency_score * 0.30
    )

    if local_text.detected_sensitive_phrases:
        score -= min(
            20.0,
            len(local_text.detected_sensitive_phrases) * 6.0,
        )

    return max(0.0, min(100.0, score))


def _llm_copy_score(
    llm: LLMReview,
    fallback: float,
) -> float:
    return _first_number(
        llm.copy_review,
        (
            "copy_quality_score",
            "clarity_score",
            "overall_score",
            "score",
        ),
        fallback,
    )


def _llm_trust_score(
    llm: LLMReview,
    fallback: float,
) -> float:
    return _first_number(
        llm.copy_review,
        (
            "trust_score",
            "transparency_score",
            "overall_score",
            "score",
        ),
        fallback,
    )


def _llm_judge_score(
    llm: LLMReview,
    fallback: float,
) -> float:
    return _first_number(
        llm.judge_review,
        (
            "holistic_score",
            "overall_score",
            "evidence_score",
            "score",
        ),
        fallback,
    )


def _build_component_explanations(
    ad: AdInput,
    local_text: LocalTextSignals,
) -> dict[str, str]:
    explanations = {
        "relevance": (
            "CLIP-ViT alignment between the listing story and "
            "the uploaded images."
        ),
        "cohesion": (
            "Whether the photos appear to show the same item "
            "without unrelated outliers."
        ),
        "quality": (
            "Local measurements of resolution, exposure, and "
            "sharpness."
        ),
        "copy": (
            "Title, description, category, location, price, "
            "condition, and usage information."
        ),
        "completeness": (
            "Coverage of seller-provided fields such as defects, "
            "accessories, ownership, and reason for selling."
        ),
        "transparency": (
            "Explicit disclosure of defects, repairs, included "
            "items, receipt, warranty, and inspection options."
        ),
        "transaction": (
            "Clarity of payment, delivery, viewing, and "
            "transaction conditions."
        ),
        "judge": (
            "Heavy multimodal reviewer, or a local evidence proxy "
            "when offline."
        ),
    }

    if ad.viewing_available:
        explanations["transaction"] += (
            " The seller offers viewing or testing."
        )

    if local_text.detected_sensitive_phrases:
        explanations["transaction"] += (
            " Sensitive transaction language reduced this score."
        )

    return explanations


class ScoreComposer:
    """
    Combines model evidence without allowing one model to own the score.

    Final score dimensions:

    - Text ↔ image relevance: 25%
    - Image-set cohesion: 10%
    - Photo quality: 10%
    - Copy quality: 15%
    - Seller information completeness: 10%
    - Transparency: 10%
    - Transaction trust: 10%
    - Evidence judge: 10%

    LLM output is blended into individual dimensions but cannot
    independently determine the final score.
    """

    def compose(
        self,
        ad: AdInput,
        clip: ClipSignals,
        image: ImageSignals,
        local_text: LocalTextSignals,
        llm: LLMReview,
        image_count: int,
    ) -> AnalysisResult:
        relevance_score = _clip_relevance_score(clip)
        cohesion_score = _cohesion_score(clip)
        quality_score = _quality_score(image)

        copy_score = _completeness_score(local_text)
        completeness_score = _seller_context_score(
            ad,
            local_text,
        )
        transparency_score = _transparency_score(local_text)
        transaction_score = _transaction_score(
            ad,
            local_text,
        )

        judge_proxy_score = (
            relevance_score * 0.35
            + cohesion_score * 0.15
            + quality_score * 0.10
            + copy_score * 0.15
            + completeness_score * 0.10
            + transparency_score * 0.10
            + transaction_score * 0.05
        )

        judge_score = judge_proxy_score

        if llm.enabled:
            copy_score = (
                copy_score * 0.60
                + _llm_copy_score(llm, copy_score) * 0.40
            )

            transparency_score = (
                transparency_score * 0.60
                + _llm_trust_score(
                    llm,
                    transparency_score,
                )
                * 0.40
            )

            judge_score = (
                judge_proxy_score * 0.45
                + _llm_judge_score(
                    llm,
                    judge_proxy_score,
                )
                * 0.55
            )

        explanations = _build_component_explanations(
            ad,
            local_text,
        )

        components = [
            ScoreComponent(
                key="relevance",
                label="Text ↔ image relevance",
                score=relevance_score,
                weight=0.25,
                explanation=explanations["relevance"],
            ),
            ScoreComponent(
                key="cohesion",
                label="Image-set cohesion",
                score=cohesion_score,
                weight=0.10,
                explanation=explanations["cohesion"],
            ),
            ScoreComponent(
                key="quality",
                label="Photo quality",
                score=quality_score,
                weight=0.10,
                explanation=explanations["quality"],
            ),
            ScoreComponent(
                key="copy",
                label="Ad copy quality",
                score=copy_score,
                weight=0.15,
                explanation=explanations["copy"],
            ),
            ScoreComponent(
                key="completeness",
                label="Seller information completeness",
                score=completeness_score,
                weight=0.10,
                explanation=explanations["completeness"],
            ),
            ScoreComponent(
                key="transparency",
                label="Defect transparency",
                score=transparency_score,
                weight=0.10,
                explanation=explanations["transparency"],
            ),
            ScoreComponent(
                key="transaction",
                label="Transaction clarity",
                score=transaction_score,
                weight=0.10,
                explanation=explanations["transaction"],
            ),
            ScoreComponent(
                key="judge",
                label="Evidence judge",
                score=judge_score,
                weight=0.10,
                explanation=explanations["judge"],
            ),
        ]

        score = sum(
            component.contribution
            for component in components
        )

        if clip.near_duplicate_pairs:
            duplicate_penalty = min(
                6.0,
                2.0 * len(clip.near_duplicate_pairs),
            )
            score -= duplicate_penalty

        if clip.outlier_indices:
            outlier_penalty = min(
                8.0,
                2.0 * len(clip.outlier_indices),
            )
            score -= outlier_penalty

        score = round(
            max(0.0, min(100.0, score)),
            1,
        )

        strengths: list[str] = []
        risks: list[str] = []
        recommendations: list[str] = []

        risks.extend(local_text.warnings)
        recommendations.extend(local_text.suggestions)
        recommendations.extend(image.warnings)

        if relevance_score >= 75:
            strengths.append(
                "The photos strongly match what the ad claims to sell."
            )
        elif relevance_score < 48:
            risks.append(
                "The text and images have weak semantic alignment."
            )
            recommendations.append(
                "Rewrite the title and description around what is "
                "visibly shown, or replace unrelated photos."
            )

        if cohesion_score >= 78 and image_count > 1:
            strengths.append(
                "The image set is visually consistent across angles."
            )
        elif cohesion_score < 50 and image_count > 1:
            risks.append(
                "The uploaded photos may not consistently represent "
                "the same item."
            )
            recommendations.append(
                "Use photos from the same item and remove unrelated "
                "or stock images."
            )

        if quality_score >= 75:
            strengths.append(
                "The photos are clear, well exposed, and large enough "
                "to inspect."
            )
        elif quality_score < 50:
            risks.append(
                "Photo quality may make it difficult for buyers to "
                "inspect the item."
            )
            recommendations.append(
                "Upload sharper, brighter photos with useful close-ups."
            )

        if copy_score >= 80:
            strengths.append(
                "The title and description provide useful buyer context."
            )

        if completeness_score >= 75:
            strengths.append(
                "The seller provided most of the structured information "
                "buyers typically need."
            )
        elif completeness_score < 50:
            risks.append(
                "Several seller-information fields are missing."
            )
            recommendations.append(
                "Complete the condition, usage history, defects, "
                "accessories, and transaction fields."
            )

        if transparency_score >= 75:
            strengths.append(
                "The listing contains useful transparency signals."
            )
        elif transparency_score < 45:
            risks.append(
                "The listing provides limited information about defects "
                "or product history."
            )
            recommendations.append(
                "Explicitly describe defects, repairs, missing accessories, "
                "and warranty status."
            )

        if transaction_score >= 75:
            strengths.append(
                "The transaction and inspection conditions are relatively clear."
            )
        elif transaction_score < 45:
            risks.append(
                "The transaction conditions are not sufficiently clear."
            )
            recommendations.append(
                "Specify the payment method, delivery method, and whether "
                "the buyer can inspect or test the item."
            )

        if clip.outlier_indices:
            labels = ", ".join(
                str(index + 1)
                for index in clip.outlier_indices
            )

            risks.append(
                f"Possible unrelated or weakly relevant image(s): {labels}."
            )

            recommendations.append(
                "Remove the outlier photo or explain why it belongs "
                "to the listing."
            )

        if clip.near_duplicate_pairs:
            pairs = ", ".join(
                f"{first + 1}↔{second + 1}"
                for first, second in clip.near_duplicate_pairs
            )

            risks.append(
                "Near-duplicate image pairs add little new evidence: "
                f"{pairs}."
            )

            recommendations.append(
                "Replace duplicate shots with the back, sides, serial "
                "label, accessories, or defects."
            )

        if local_text.detected_sensitive_phrases:
            risks.append(
                "The text contains potentially high-pressure or risky "
                "transaction language."
            )

        if llm.enabled:
            strengths.extend(
                _strings(
                    llm.judge_review.get("strengths")
                )
            )

            risks.extend(
                _strings(
                    llm.judge_review.get("risks")
                )
            )

            recommendations.extend(
                _strings(
                    llm.judge_review.get("recommendations")
                )
            )

        confidence = self._confidence(
            clip=clip,
            image=image,
            local_text=local_text,
            llm=llm,
            image_count=image_count,
        )

        if score >= 85:
            verdict = (
                "Excellent — credible, coherent, and ready to publish"
            )
        elif score >= 70:
            verdict = (
                "Strong — a few focused edits could improve conversion"
            )
        elif score >= 50:
            verdict = (
                "Needs work — buyers may hesitate or misunderstand "
                "the offer"
            )
        else:
            verdict = (
                "Weak — major relevance, quality, or trust issues detected"
            )

        metadata = {
            "image_count": image_count,
            "score_version": "seller-context-v2",
            "seller_context": {
                "condition": ad.condition,
                "usage_duration": ad.usage_duration,
                "manufacture_year": ad.manufacture_year,
                "ownership_status": ad.ownership_status,
                "seller_type": ad.seller_type,
                "defects": ad.defects,
                "included_items": ad.included_items,
                "reason_for_selling": ad.reason_for_selling,
                "seller_note": ad.seller_note,
                "transaction_method": ad.transaction_method,
                "delivery_options": ad.delivery_options,
                "negotiable": ad.negotiable,
                "exchange_possible": ad.exchange_possible,
                "viewing_available": ad.viewing_available,
                "original_receipt_available": (
                    ad.original_receipt_available
                ),
                "warranty_available": ad.warranty_available,
            },
            "score_inputs": {
                "relevance": round(relevance_score, 2),
                "cohesion": round(cohesion_score, 2),
                "quality": round(quality_score, 2),
                "copy": round(copy_score, 2),
                "completeness": round(
                    completeness_score,
                    2,
                ),
                "transparency": round(
                    transparency_score,
                    2,
                ),
                "transaction": round(
                    transaction_score,
                    2,
                ),
                "judge": round(judge_score, 2),
            },
            "local_text": {
                "completeness_score": (
                    local_text.completeness_score
                ),
                "trust_score": local_text.trust_score,
                "transparency_score": (
                    local_text.transparency_score
                ),
                "detected_sensitive_phrases": (
                    local_text.detected_sensitive_phrases
                ),
            },
        }

        return AnalysisResult(
            final_score=score,
            confidence=confidence,
            verdict=verdict,
            components=components,
            clip=clip,
            image=image,
            local_text=local_text,
            llm=llm,
            strengths=_dedupe(strengths)[:8],
            risks=_dedupe(risks)[:10],
            recommendations=_dedupe(
                recommendations
            )[:10],
            metadata=metadata,
        )

    @staticmethod
    def _confidence(
        clip: ClipSignals,
        image: ImageSignals,
        local_text: LocalTextSignals,
        llm: LLMReview,
        image_count: int,
    ) -> float:
        confidence = 35.0

        if image_count >= 3:
            confidence += 18.0
        elif image_count == 2:
            confidence += 10.0
        elif image_count == 1:
            confidence += 5.0

        if (
            len(clip.per_image_relevance) == image_count
            and image_count > 0
        ):
            confidence += 12.0

        if (
            len(image.per_image_quality) == image_count
            and image_count > 0
        ):
            confidence += 8.0

        if local_text.completeness_score >= 70:
            confidence += 8.0

        if local_text.transparency_score >= 60:
            confidence += 5.0

        if llm.enabled:
            confidence += 12.0

        if llm.errors:
            confidence -= min(
                18.0,
                len(llm.errors) * 5.0,
            )

        if clip.outlier_indices:
            confidence -= min(
                8.0,
                len(clip.outlier_indices) * 2.0,
            )

        return round(
            max(10.0, min(96.0, confidence)),
            1,
        )