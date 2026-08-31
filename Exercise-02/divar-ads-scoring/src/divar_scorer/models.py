from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from PIL import Image


@dataclass(slots=True)
class AdInput:
    """
    All information provided by the seller for one marketplace listing.
    """

    title: str
    description: str

    category: str = ""
    city: str = ""
    price: str = ""

    # Basic seller-provided context
    condition: str = "نامشخص"
    usage_duration: str = ""
    manufacture_year: int | None = None
    ownership_status: str = "نامشخص"
    seller_type: str = "شخصی"

    # Transparency and product details
    defects: str = ""
    included_items: str = ""
    reason_for_selling: str = ""
    seller_note: str = ""

    # Transaction information
    transaction_method: list[str] = field(default_factory=list)
    delivery_options: list[str] = field(default_factory=list)

    negotiable: bool = False
    exchange_possible: bool = False
    viewing_available: bool = False
    original_receipt_available: bool = False
    warranty_available: bool = False

    # Runtime-only image objects
    images: list["Image.Image"] = field(
        default_factory=list,
        repr=False,
    )
    image_names: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """
        Text representation used by local text analysis,
        CLIP caption generation, and LLM prompts.
        """

        fields = [
            f"Title: {self.title.strip()}",
            f"Category: {self.category.strip()}",
            f"Location: {self.city.strip()}",
            f"Price: {self.price.strip()}",
            f"Condition: {self.condition.strip()}",
            f"Usage duration: {self.usage_duration.strip()}",
            f"Manufacture year: {self.manufacture_year or ''}",
            f"Ownership status: {self.ownership_status.strip()}",
            f"Seller type: {self.seller_type.strip()}",
            f"Defects and repairs: {self.defects.strip()}",
            f"Included items: {self.included_items.strip()}",
            f"Reason for selling: {self.reason_for_selling.strip()}",
            f"Seller note: {self.seller_note.strip()}",
            (
                "Transaction methods: "
                + ", ".join(self.transaction_method)
            ),
            (
                "Delivery options: "
                + ", ".join(self.delivery_options)
            ),
            (
                "Negotiable price: "
                + ("yes" if self.negotiable else "no")
            ),
            (
                "Exchange possible: "
                + ("yes" if self.exchange_possible else "no")
            ),
            (
                "Viewing or testing available: "
                + ("yes" if self.viewing_available else "no")
            ),
            (
                "Original receipt available: "
                + (
                    "yes"
                    if self.original_receipt_available
                    else "no"
                )
            ),
            (
                "Warranty available: "
                + ("yes" if self.warranty_available else "no")
            ),
            f"Description: {self.description.strip()}",
        ]

        return "\n".join(
            item
            for item in fields
            if item.split(":", 1)[1].strip()
        )

    @property
    def seller_context(self) -> str:
        """
        Compact structured context for LLM prompts.
        """

        return "\n".join(
            [
                f"Condition: {self.condition.strip()}",
                f"Usage duration: {self.usage_duration.strip()}",
                f"Manufacture year: {self.manufacture_year or 'unknown'}",
                f"Ownership status: {self.ownership_status.strip()}",
                f"Seller type: {self.seller_type.strip()}",
                f"Defects: {self.defects.strip() or 'not provided'}",
                (
                    "Included items: "
                    f"{self.included_items.strip() or 'not provided'}"
                ),
                (
                    "Reason for selling: "
                    f"{self.reason_for_selling.strip() or 'not provided'}"
                ),
                (
                    "Transaction methods: "
                    f"{', '.join(self.transaction_method) or 'not provided'}"
                ),
                (
                    "Delivery options: "
                    f"{', '.join(self.delivery_options) or 'not provided'}"
                ),
                (
                    "Negotiable: "
                    f"{'yes' if self.negotiable else 'no'}"
                ),
                (
                    "Exchange possible: "
                    f"{'yes' if self.exchange_possible else 'no'}"
                ),
                (
                    "Viewing available: "
                    f"{'yes' if self.viewing_available else 'no'}"
                ),
                (
                    "Original receipt available: "
                    f"{'yes' if self.original_receipt_available else 'no'}"
                ),
                (
                    "Warranty available: "
                    f"{'yes' if self.warranty_available else 'no'}"
                ),
                f"Seller note: {self.seller_note.strip() or 'none'}",
            ]
        )


@dataclass(slots=True)
class ClipSignals:
    text_image_score: float = 0.0
    image_cohesion_score: float = 0.0

    per_image_relevance: list[float] = field(
        default_factory=list
    )

    outlier_indices: list[int] = field(
        default_factory=list
    )

    near_duplicate_pairs: list[tuple[int, int]] = field(
        default_factory=list
    )

    raw_text_image_cosines: list[float] = field(
        default_factory=list
    )

    visual_caption_used: str = ""


@dataclass(slots=True)
class ImageSignals:
    quality_score: float = 0.0

    per_image_quality: list[float] = field(
        default_factory=list
    )

    resolution_scores: list[float] = field(
        default_factory=list
    )

    exposure_scores: list[float] = field(
        default_factory=list
    )

    sharpness_scores: list[float] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )


@dataclass(slots=True)
class LocalTextSignals:
    completeness_score: float = 0.0
    trust_score: float = 0.0

    # Existing names used by your current UI or scoring logic
    warnings: list[str] = field(
        default_factory=list
    )

    suggestions: list[str] = field(
        default_factory=list
    )

    # Additional explainability data
    title_score: float = 0.0
    description_score: float = 0.0
    transparency_score: float = 0.0
    consistency_score: float = 0.0

    detected_sensitive_phrases: list[str] = field(
        default_factory=list
    )

    has_contact_info: bool = False
    has_defects: bool = False
    has_included_items: bool = False
    viewing_available: bool = False
    warranty_available: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class LLMReview:
    enabled: bool = False

    models_used: list[str] = field(
        default_factory=list
    )

    visual_brief: dict[str, Any] = field(
        default_factory=dict
    )

    copy_review: dict[str, Any] = field(
        default_factory=dict
    )

    judge_review: dict[str, Any] = field(
        default_factory=dict
    )

    errors: list[str] = field(
        default_factory=list
    )


@dataclass(slots=True)
class ScoreComponent:
    key: str
    label: str
    score: float
    weight: float
    explanation: str = ""

    @property
    def contribution(self) -> float:
        return self.score * self.weight


@dataclass(slots=True)
class AnalysisResult:
    final_score: float
    confidence: float
    verdict: str

    components: list[ScoreComponent]

    clip: ClipSignals
    image: ImageSignals
    local_text: LocalTextSignals
    llm: LLMReview

    strengths: list[str] = field(
        default_factory=list
    )

    risks: list[str] = field(
        default_factory=list
    )

    recommendations: list[str] = field(
        default_factory=list
    )

    # Stores seller context and audit information
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the complete nested dataclass tree to a dictionary.

        Images are excluded because AdInput is not stored inside
        AnalysisResult. All nested dataclasses are recursively converted.
        """

        return asdict(self)