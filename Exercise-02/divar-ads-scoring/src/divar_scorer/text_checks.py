from __future__ import annotations

import re

from .models import AdInput, LocalTextSignals


_PHONE_RE = re.compile(
    r"(?<!\d)(?:(?:\+?98|0098|0)?9\d{9})(?!\d)"
)

_URL_RE = re.compile(
    r"(?:https?://|www\.|[a-zA-Z0-9-]+\.(?:com|ir|net|org)\b)",
    re.IGNORECASE,
)

_PERSIAN_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)

_PRESSURE_TERMS = (
    "فقط امروز",
    "فقط تا امروز",
    "فوری",
    "تماس فوری",
    "بیعانه",
    "کارت به کارت",
    "واریز قبل از بازدید",
    "واریز قبل از تست",
    "پرداخت قبل از بازدید",
    "پرداخت قبل از تست",
    "اول واریز",
    "بدون تست",
    "urgent transfer",
    "deposit first",
    "whatsapp only",
    "pay before viewing",
    "transfer before inspection",
)

_UNKNOWN_CONDITION = {
    "",
    "نامشخص",
    "unknown",
    "not specified",
}

_UNKNOWN_OWNERSHIP = {
    "",
    "نامشخص",
    "unknown",
    "not specified",
}

_UNKNOWN_SELLER_TYPE = {
    "",
    "نامشخص",
    "unknown",
    "not specified",
}


def _normalized_text(value: str) -> str:
    """
    Normalize whitespace and Persian/Arabic digits.
    """

    normalized = value.translate(_PERSIAN_DIGITS)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _has_value(value: object) -> bool:
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, (list, tuple, set)):
        return bool(value)

    return True


def _is_known(value: str, unknown_values: set[str]) -> bool:
    return value.strip().casefold() not in unknown_values


def _contains_phone(text: str) -> bool:
    """
    Detect common Iranian mobile-number formats after digit normalization.
    """

    normalized = _normalized_text(text)
    compact = re.sub(r"[\s\-().]", "", normalized)

    return bool(_PHONE_RE.search(compact))


def _contains_url(text: str) -> bool:
    return bool(_URL_RE.search(text))


def _contains_pressure_language(text: str) -> list[str]:
    normalized = text.casefold()

    return [
        term
        for term in _PRESSURE_TERMS
        if term.casefold() in normalized
    ]


def _uppercase_ratio(text: str) -> float:
    latin_letters = [
        char
        for char in text
        if char.isascii() and char.isalpha()
    ]

    if not latin_letters:
        return 0.0

    uppercase_count = sum(
        char.isupper()
        for char in latin_letters
    )

    return uppercase_count / len(latin_letters)


def _word_count(text: str) -> int:
    return len(
        [
            word
            for word in re.split(r"\s+", text)
            if word.strip()
        ]
    )


def _unique_word_ratio(text: str) -> float:
    words = [
        word.casefold()
        for word in re.findall(r"\S+", text)
        if word.strip()
    ]

    if not words:
        return 0.0

    return len(set(words)) / len(words)


def _calculate_field_completeness(ad: AdInput) -> float:
    """
    Calculate completeness of seller-provided structured information.

    The weights intentionally reward useful information while avoiding
    the assumption that every optional field must be filled.
    """

    weighted_fields: list[tuple[bool, float]] = [
        (bool(ad.title.strip()), 10.0),
        (bool(ad.description.strip()), 20.0),
        (bool(ad.category.strip()), 8.0),
        (bool(ad.city.strip()), 6.0),
        (bool(ad.price.strip()), 8.0),

        (
            _is_known(
                ad.condition,
                _UNKNOWN_CONDITION,
            ),
            10.0,
        ),
        (bool(ad.usage_duration.strip()), 6.0),
        (ad.manufacture_year is not None, 4.0),

        (
            _is_known(
                ad.ownership_status,
                _UNKNOWN_OWNERSHIP,
            ),
            4.0,
        ),
        (
            _is_known(
                ad.seller_type,
                _UNKNOWN_SELLER_TYPE,
            ),
            4.0,
        ),

        (bool(ad.defects.strip()), 7.0),
        (bool(ad.included_items.strip()), 5.0),
        (bool(ad.reason_for_selling.strip()), 3.0),

        (bool(ad.transaction_method), 2.5),
        (bool(ad.delivery_options), 2.5),
    ]

    total_weight = sum(
        weight
        for _, weight in weighted_fields
    )

    earned_weight = sum(
        weight
        for present, weight in weighted_fields
        if present
    )

    if total_weight == 0:
        return 0.0

    return max(
        0.0,
        min(
            100.0,
            earned_weight / total_weight * 100.0,
        ),
    )


def _calculate_transparency_score(ad: AdInput) -> float:
    """
    Score explicit disclosure of information that reduces buyer hesitation.

    A missing optional field is not automatically treated as deception.
    It only means that the listing gives the buyer less evidence.
    """

    transparency_items: list[tuple[bool, float]] = [
        (bool(ad.defects.strip()), 30.0),
        (bool(ad.included_items.strip()), 15.0),
        (bool(ad.reason_for_selling.strip()), 10.0),
        (ad.viewing_available, 15.0),
        (ad.original_receipt_available, 15.0),
        (ad.warranty_available, 15.0),
    ]

    total_weight = sum(
        weight
        for _, weight in transparency_items
    )

    earned_weight = sum(
        weight
        for present, weight in transparency_items
        if present
    )

    if total_weight == 0:
        return 0.0

    return max(
        0.0,
        min(
            100.0,
            earned_weight / total_weight * 100.0,
        ),
    )


def _calculate_copy_score(
    title: str,
    description: str,
) -> float:
    title_score = min(
        100.0,
        len(title) / 30.0 * 100.0,
    )

    description_score = min(
        100.0,
        len(description) / 300.0 * 100.0,
    )

    return (
        title_score * 0.40
        + description_score * 0.60
    )


def _calculate_transaction_score(
    ad: AdInput,
    has_phone: bool,
    pressure_terms: list[str],
) -> float:
    """
    Measure transaction clarity, not seller legitimacy.

    Having a phone number is not inherently fraudulent; the small
    deduction is only for public-copy exposure and should be treated
    as a platform-safety signal.
    """

    score = 35.0

    if ad.transaction_method:
        score += 15.0

    if ad.delivery_options:
        score += 10.0

    if ad.viewing_available:
        score += 15.0

    if ad.city.strip():
        score += 5.0

    if ad.price.strip():
        score += 5.0

    if ad.original_receipt_available:
        score += 5.0

    if ad.warranty_available:
        score += 5.0

    if has_phone:
        score -= 5.0

    if pressure_terms:
        score -= min(
            30.0,
            len(pressure_terms) * 8.0,
        )

    return max(
        0.0,
        min(100.0, score),
    )


def analyze_text(ad: AdInput) -> LocalTextSignals:
    """
    Analyze listing text and seller-provided context locally.

    The result is decision support for listing quality. It is not
    an identity check, fraud verdict, appraisal, or legal conclusion.
    """

    title = ad.title.strip()
    description = ad.description.strip()

    combined = "\n".join(
        [
            title,
            description,
            ad.category.strip(),
            ad.city.strip(),
            ad.price.strip(),
            ad.condition.strip(),
            ad.usage_duration.strip(),
            ad.ownership_status.strip(),
            ad.seller_type.strip(),
            ad.defects.strip(),
            ad.included_items.strip(),
            ad.reason_for_selling.strip(),
            ad.seller_note.strip(),
            " ".join(ad.transaction_method),
            " ".join(ad.delivery_options),
        ]
    ).strip()

    warnings: list[str] = []
    suggestions: list[str] = []

    normalized_combined = _normalized_text(combined)

    completeness_score = _calculate_field_completeness(ad)
    transparency_score = _calculate_transparency_score(ad)
    copy_score = _calculate_copy_score(
        title,
        description,
    )

    has_phone = _contains_phone(normalized_combined)
    has_url = _contains_url(normalized_combined)

    pressure_terms = _contains_pressure_language(
        normalized_combined
    )

    uppercase_ratio = _uppercase_ratio(combined)
    unique_word_ratio = _unique_word_ratio(description)

    # Basic copy quality checks
    if not title:
        warnings.append(
            "The listing has no title."
        )
        suggestions.append(
            "Add a specific title with the brand, model, "
            "condition, or most important attribute."
        )
    elif len(title) < 8:
        warnings.append(
            "The title is very short."
        )
        suggestions.append(
            "Make the title more specific so buyers can understand "
            "and find the item more easily."
        )
    elif len(title) > 80:
        warnings.append(
            "The title is longer than recommended."
        )
        suggestions.append(
            "Shorten the title and keep only the most important "
            "identifying information."
        )

    if not description:
        warnings.append(
            "The listing has no description."
        )
        suggestions.append(
            "Describe condition, usage history, defects, dimensions, "
            "included items, and reason for selling."
        )
    elif len(description) < 100:
        warnings.append(
            "The description is short."
        )
        suggestions.append(
            "Add condition, age, defects, dimensions or specifications, "
            "included items, and reason for selling."
        )

    if not ad.price.strip():
        warnings.append(
            "No price was provided."
        )
        suggestions.append(
            "Add a price or clearly explain the negotiation conditions."
        )

    if len(ad.images) == 0:
        warnings.append(
            "No images were provided."
        )
        suggestions.append(
            "Upload original photos showing several useful angles."
        )
    elif len(ad.images) < 3:
        suggestions.append(
            "Use at least three original angles, including a close-up "
            "of important details or defects."
        )

    # Structured seller information checks
    if not _is_known(
        ad.condition,
        _UNKNOWN_CONDITION,
    ):
        warnings.append(
            "The product condition is not specified."
        )
        suggestions.append(
            "Select or describe the actual condition of the item."
        )

    if not ad.usage_duration.strip():
        suggestions.append(
            "Add the usage duration or current mileage/count where relevant."
        )

    if not ad.defects.strip():
        warnings.append(
            "Defects, repairs, or limitations are not explicitly described."
        )
        suggestions.append(
            "State known defects, repairs, scratches, replaced parts, "
            "or explicitly say that no known defect exists."
        )

    if not ad.included_items.strip():
        suggestions.append(
            "Specify the included accessories, packaging, documents, "
            "or missing items."
        )

    if not ad.reason_for_selling.strip():
        suggestions.append(
            "Add the reason for selling if you want to reduce buyer hesitation."
        )

    if not ad.transaction_method:
        suggestions.append(
            "Specify acceptable transaction or payment methods."
        )

    if not ad.delivery_options:
        suggestions.append(
            "Specify whether pickup, local delivery, or shipping is available."
        )

    if not ad.viewing_available:
        suggestions.append(
            "If possible, offer viewing or testing before purchase."
        )

    # Public-copy safety checks
    if has_phone:
        warnings.append(
            "A phone number appears in the public copy; keep contact "
            "inside the platform where possible."
        )

    if has_url:
        warnings.append(
            "An external link appears in the ad copy."
        )
        suggestions.append(
            "Remove unnecessary external links and keep the transaction "
            "inside trusted platform flows."
        )

    if pressure_terms:
        warnings.append(
            "The copy contains urgency or off-platform payment language: "
            + ", ".join(pressure_terms)
            + "."
        )
        suggestions.append(
            "Describe payment and timing conditions clearly without "
            "pressuring the buyer to transfer money before inspection."
        )

    if combined.count("!") >= 4:
        warnings.append(
            "Excessive exclamation marks make the ad look promotional "
            "or spam-like."
        )
        suggestions.append(
            "Use neutral wording and keep emphasis focused on factual details."
        )

    if (
        uppercase_ratio > 0.65
        and sum(
            char.isascii() and char.isalpha()
            for char in combined
        ) > 12
    ):
        warnings.append(
            "Large blocks of uppercase text reduce readability."
        )
        suggestions.append(
            "Use normal capitalization and reserve uppercase text "
            "for short labels."
        )

    if (
        len(description) > 50
        and _word_count(description) >= 8
        and unique_word_ratio < 0.45
    ):
        warnings.append(
            "The description appears repetitive."
        )
        suggestions.append(
            "Replace repeated claims with concrete specifications, "
            "history, condition details, and photos."
        )

    if (
        ad.negotiable
        and "مذاکره" not in combined.casefold()
        and "negoti" not in combined.casefold()
    ):
        suggestions.append(
            "Mention the negotiation conditions in the description "
            "because the price is marked as negotiable."
        )

    # Local trust score
    transaction_score = _calculate_transaction_score(
        ad=ad,
        has_phone=has_phone,
        pressure_terms=pressure_terms,
    )

    trust_score = (
        completeness_score * 0.25
        + transparency_score * 0.30
        + transaction_score * 0.25
        + copy_score * 0.20
    )

    if has_url:
        trust_score -= 10.0

    if uppercase_ratio > 0.65:
        trust_score -= 6.0

    if unique_word_ratio < 0.45 and len(description) > 50:
        trust_score -= 8.0

    trust_score = max(
        0.0,
        min(100.0, trust_score),
    )

    # Overall local text score
    overall_score = (
        copy_score * 0.25
        + completeness_score * 0.30
        + transparency_score * 0.20
        + trust_score * 0.25
    )

    overall_score = max(
        0.0,
        min(100.0, overall_score),
    )

    return LocalTextSignals(
        completeness_score=round(
            overall_score,
            1,
        ),
        trust_score=round(
            trust_score,
            1,
        ),
        warnings=_dedupe(warnings),
        suggestions=_dedupe(suggestions),
        title_score=round(
            min(100.0, len(title) / 30.0 * 100.0),
            1,
        ),
        description_score=round(
            min(100.0, len(description) / 300.0 * 100.0),
            1,
        ),
        transparency_score=round(
            transparency_score,
            1,
        ),
        consistency_score=0.0,
        detected_sensitive_phrases=pressure_terms,
        has_contact_info=has_phone,
        has_defects=bool(ad.defects.strip()),
        has_included_items=bool(ad.included_items.strip()),
        viewing_available=ad.viewing_available,
        warranty_available=ad.warranty_available,
        metadata={
            "copy_score": round(copy_score, 1),
            "field_completeness_score": round(
                completeness_score,
                1,
            ),
            "transparency_score": round(
                transparency_score,
                1,
            ),
            "transaction_score": round(
                transaction_score,
                1,
            ),
            "title_length": len(title),
            "description_length": len(description),
            "image_count": len(ad.images),
            "has_phone": has_phone,
            "has_external_url": has_url,
            "pressure_terms": pressure_terms,
            "uppercase_ratio": round(
                uppercase_ratio,
                3,
            ),
            "unique_word_ratio": round(
                unique_word_ratio,
                3,
            ),
            "condition_provided": _is_known(
                ad.condition,
                _UNKNOWN_CONDITION,
            ),
            "usage_duration_provided": bool(
                ad.usage_duration.strip()
            ),
            "manufacture_year_provided": (
                ad.manufacture_year is not None
            ),
            "ownership_status_provided": _is_known(
                ad.ownership_status,
                _UNKNOWN_OWNERSHIP,
            ),
            "seller_type_provided": _is_known(
                ad.seller_type,
                _UNKNOWN_SELLER_TYPE,
            ),
            "transaction_methods_provided": bool(
                ad.transaction_method
            ),
            "delivery_options_provided": bool(
                ad.delivery_options
            ),
        },
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for item in items:
        normalized = item.strip().casefold()

        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(item.strip())

    return output