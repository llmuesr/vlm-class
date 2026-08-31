from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from divar_scorer.clip_engine import ClipAnalyzer  # noqa: E402
from divar_scorer.config import Settings  # noqa: E402
from divar_scorer.image_quality import analyze_image_quality  # noqa: E402
from divar_scorer.models import AdInput, AnalysisResult, LLMReview  # noqa: E402
from divar_scorer.openrouter import OpenRouterJury  # noqa: E402
from divar_scorer.scoring import ScoreComposer  # noqa: E402
from divar_scorer.text_checks import analyze_text  # noqa: E402


st.set_page_config(
    page_title="DivarLens — Ad Quality Score",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');

:root {
  --brick: #a62626;
  --ink: #25211f;
  --muted: #766f6a;
  --sand: #f7f5f2;
  --line: #e7e0da;
}

html, body, [class*="css"] {
  font-family: "Vazirmatn", sans-serif;
}

.stApp {
  background:
    radial-gradient(circle at 92% 3%, #f0dedd 0, transparent 28%),
    var(--sand);
}

.block-container {
  max-width: 1240px;
  padding-top: 2.2rem;
  padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
  background: #211e1c;
  color: #f8f4ef;
}

[data-testid="stSidebar"] * {
  color: #f8f4ef;
}

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
  color: #25211f !important;
}

.hero {
  background: linear-gradient(125deg, #231f1d 0%, #3b2c28 62%, #7e2424 100%);
  color: #fff;
  border-radius: 24px;
  padding: 34px 38px;
  margin-bottom: 22px;
  box-shadow: 0 18px 50px rgba(60, 30, 24, .16);
}

.hero-kicker {
  color: #efbbb6;
  font-size: .78rem;
  letter-spacing: .16em;
  text-transform: uppercase;
  font-weight: 700;
}

.hero h1 {
  font-size: 2.5rem;
  line-height: 1.05;
  margin: .45rem 0 .7rem;
  letter-spacing: -.035em;
}

.hero p {
  color: #e7ddd8;
  max-width: 760px;
  font-size: 1.03rem;
  margin: 0;
}

.micro-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 18px;
}

.micro {
  border: 1px solid rgba(255, 255, 255, .18);
  border-radius: 999px;
  padding: 6px 11px;
  font-size: .76rem;
  color: #f4e9e5;
  background: rgba(255, 255, 255, .05);
}

.section-card {
  background: rgba(255, 255, 255, .86);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 20px 22px;
  box-shadow: 0 8px 26px rgba(55, 40, 34, .055);
  margin: 8px 0 18px;
}

.score-shell {
  display: flex;
  align-items: center;
  gap: 28px;
  padding: 12px 4px 16px;
}

.score-ring {
  width: 170px;
  height: 170px;
  min-width: 170px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  position: relative;
  background: conic-gradient(
    var(--ring) calc(var(--score) * 1%),
    #eadfda 0
  );
}

.score-ring:after {
  content: "";
  position: absolute;
  inset: 12px;
  border-radius: 50%;
  background: #fffaf7;
}

.score-number {
  position: relative;
  z-index: 2;
  text-align: center;
  line-height: 1;
}

.score-number strong {
  font-size: 3.45rem;
  letter-spacing: -.07em;
  color: var(--ink);
}

.score-number span {
  display: block;
  color: var(--muted);
  font-size: .75rem;
  margin-top: 8px;
  letter-spacing: .08em;
}

.score-copy h2 {
  margin: 0 0 6px;
  font-size: 1.45rem;
}

.score-copy p {
  margin: 0 0 12px;
  color: var(--muted);
}

.confidence {
  display: inline-block;
  background: #f0e7e2;
  border-radius: 999px;
  padding: 6px 11px;
  font-size: .78rem;
}

.component {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 15px;
  padding: 14px 15px;
  min-height: 132px;
}

.component-label {
  font-size: .78rem;
  color: var(--muted);
  min-height: 38px;
}

.component-score {
  font-size: 1.8rem;
  font-weight: 800;
  letter-spacing: -.04em;
  margin: 4px 0;
}

.component-weight {
  color: var(--brick);
  font-size: .72rem;
  font-weight: 600;
}

.verdict-good {
  border-left: 4px solid #287a56;
}

.verdict-warn {
  border-left: 4px solid #d18a22;
}

.verdict-risk {
  border-left: 4px solid #a62626;
}

.rtl input,
.rtl textarea,
[data-testid="stTextArea"] textarea {
  direction: rtl;
  text-align: right;
}

.tiny {
  color: var(--muted);
  font-size: .78rem;
}

.stButton > button,
.stDownloadButton > button {
  border-radius: 12px;
  font-weight: 700;
}

.stButton > button[kind="primary"] {
  background: var(--brick);
  border-color: var(--brick);
}

@media (max-width: 700px) {
  .hero {
    padding: 25px 23px;
  }

  .hero h1 {
    font-size: 2rem;
  }

  .score-shell {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)


CATEGORY_VISUAL_HINTS = {
    "وسایل نقلیه / Vehicles": (
        "the vehicle offered for sale, exterior and interior details"
    ),
    "املاک / Real estate": (
        "the offered property, its rooms, fixtures, and condition"
    ),
    "موبایل و تبلت / Phones": (
        "the phone or tablet offered for sale and its visible condition"
    ),
    "لوازم الکترونیکی / Electronics": (
        "the electronic device offered for sale and included accessories"
    ),
    "خانه و آشپزخانه / Home": (
        "the household item or furniture offered for sale"
    ),
    "مد و پوشاک / Fashion": (
        "the clothing or fashion item offered for sale"
    ),
    "خدمات / Services": (
        "visual evidence relevant to the advertised service"
    ),
    "سایر / Other": (
        "the exact item offered in a person-to-person marketplace listing"
    ),
}


def _fallback_visual_caption(ad: AdInput) -> str:
    hint = CATEGORY_VISUAL_HINTS.get(
        ad.category,
        CATEGORY_VISUAL_HINTS["سایر / Other"],
    )

    return f"A clear marketplace listing photo showing {hint}. {ad.title.strip()}"[
        :500
    ]


def _build_ad_context(ad: AdInput) -> dict[str, Any]:
    return {
        "title": ad.title,
        "description": ad.description,
        "category": ad.category,
        "city": ad.city,
        "price": ad.price,
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
        "original_receipt_available": ad.original_receipt_available,
        "warranty_available": ad.warranty_available,
    }


def _load_images(
    uploaded_files: list,
) -> tuple[list[Image.Image], list[str], list[str]]:
    images: list[Image.Image] = []
    names: list[str] = []
    errors: list[str] = []

    for uploaded in uploaded_files[:10]:
        try:
            uploaded.seek(0)
            image = Image.open(uploaded)
            image.load()

            images.append(image.convert("RGB"))
            names.append(uploaded.name)

        except (
            Image.DecompressionBombError,
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            errors.append(f"{uploaded.name}: {exc}")

    return images, names, errors


@st.cache_resource(show_spinner=False)
def _clip_analyzer(model_id: str) -> ClipAnalyzer:
    return ClipAnalyzer(model_id)


def _score_color(score: float) -> str:
    if score >= 80:
        return "#287a56"

    if score >= 60:
        return "#d18a22"

    return "#a62626"


def _render_components(result: AnalysisResult) -> None:
    columns = st.columns(3)

    for index, component in enumerate(result.components):
        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="component">
                  <div class="component-label">{component.label}</div>
                  <div class="component-score"
                       style="color:{_score_color(component.score)}">
                    {component.score:.0f}
                  </div>
                  <div class="component-weight">
                    {component.weight * 100:.0f}% of final score
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_findings(result: AnalysisResult) -> None:
    left, middle, right = st.columns(3)

    with left:
        st.subheader("What works")

        if result.strengths:
            for item in result.strengths:
                st.success(item, icon="✅")
        else:
            st.caption("No strong positive signal yet.")

    with middle:
        st.subheader("Buyer hesitation")

        if result.risks:
            for item in result.risks:
                st.warning(item, icon="⚠️")
        else:
            st.caption("No obvious risk flags detected.")

    with right:
        st.subheader("Best next edits")

        if result.recommendations:
            for number, item in enumerate(
                result.recommendations,
                start=1,
            ):
                st.markdown(f"**{number}.** {item}")
        else:
            st.caption("This ad is already in strong shape.")


def _render_image_evidence(
    result: AnalysisResult,
    images: list[Image.Image],
    names: list[str],
) -> None:
    if not images:
        return

    st.subheader("Photo-by-photo evidence")
    columns = st.columns(min(4, len(images)))

    for index, image in enumerate(images):
        with columns[index % len(columns)]:
            st.image(image, use_container_width=True)

            relevance = (
                result.clip.per_image_relevance[index]
                if index < len(result.clip.per_image_relevance)
                else 0
            )

            quality = (
                result.image.per_image_quality[index]
                if index < len(result.image.per_image_quality)
                else 0
            )

            flag = (
                " · possible outlier"
                if index in result.clip.outlier_indices
                else ""
            )

            st.markdown(f"**{index + 1}. {names[index]}**")
            st.caption(
                f"Relevance {relevance:.0f} · Quality {quality:.0f}{flag}"
            )


def _render_seller_context(result: AnalysisResult) -> None:
    ad_context = result.metadata.get("ad_context")

    if not ad_context:
        st.caption("No seller context was stored.")
        return

    st.json(ad_context)


def _render_result(
    result: AnalysisResult,
    images: list[Image.Image],
    names: list[str],
) -> None:
    score_class = (
        "verdict-good"
        if result.final_score >= 70
        else (
            "verdict-warn"
            if result.final_score >= 50
            else "verdict-risk"
        )
    )

    st.markdown(
        f'<div class="section-card {score_class}">',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="score-shell">
          <div class="score-ring"
               style="--score:{result.final_score};
                      --ring:{_score_color(result.final_score)}">
            <div class="score-number">
              <strong>{result.final_score:.0f}</strong>
              <span>OUT OF 100</span>
            </div>
          </div>

          <div class="score-copy">
            <h2>{result.verdict}</h2>
            <p>
              A blended evidence score: visual semantics, consistency,
              photo craft, copy, trust, and expert review.
            </p>
            <span class="confidence">
              {result.confidence:.0f}% evidence confidence
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Score anatomy")
    _render_components(result)

    st.markdown("<br>", unsafe_allow_html=True)
    _render_findings(result)

    st.divider()
    _render_image_evidence(result, images, names)

    with st.expander("Seller-provided context"):
        _render_seller_context(result)

    with st.expander("Model evidence & audit trail"):
        st.markdown(
            f"**CLIP visual query**  \n"
            f"{result.clip.visual_caption_used}"
        )

        st.write(
            {
                "CLIP model": st.session_state.get(
                    "clip_model_used",
                    "",
                ),
                "raw cosine similarities": (
                    result.clip.raw_text_image_cosines
                ),
                "near-duplicate pairs (0-based)": (
                    result.clip.near_duplicate_pairs
                ),
                "OpenRouter models": result.llm.models_used,
                "OpenRouter errors": result.llm.errors,
            }
        )

        if result.llm.visual_brief:
            st.markdown("**Fast visual specialist**")
            st.json(result.llm.visual_brief)

        if result.llm.copy_review:
            st.markdown("**Fast copy specialist**")
            st.json(result.llm.copy_review)

        if result.llm.judge_review:
            st.markdown("**Heavy evidence judge**")
            st.json(result.llm.judge_review)

    report_json = json.dumps(
        result.to_dict(),
        ensure_ascii=False,
        indent=2,
    )

    st.download_button(
        "Download score report (JSON)",
        data=report_json,
        file_name="divarlens-score.json",
        mime="application/json",
        use_container_width=True,
    )


with st.sidebar:
    st.markdown("## ◉ DivarLens")
    st.caption("Multimodal ad quality lab")
    st.divider()

    analysis_mode = st.radio(
        "Review mode",
        ["Deep jury", "Local-only preview"],
        help=(
            "Deep jury runs two light specialists and one heavy "
            "multimodal judge through OpenRouter."
        ),
    )

    settings = Settings()

    entered_key = st.text_input(
        "OpenRouter API key",
        value="",
        type="password",
        placeholder="Uses OPENROUTER_API_KEY if empty",
        disabled=analysis_mode != "Deep jury",
    )

    if entered_key.strip():
        settings.openrouter_api_key = entered_key.strip()

    with st.expander("Model routing"):
        light_models = st.text_area(
            "Light models (comma-separated)",
            value=", ".join(settings.light_models),
            height=84,
        )

        heavy_models = st.text_area(
            "Heavy fallbacks (comma-separated)",
            value=", ".join(settings.heavy_models),
            height=84,
        )

        clip_model = st.text_input(
            "CLIP-ViT checkpoint",
            value=settings.clip_model_id,
        )

        settings.light_models = [
            item.strip()
            for item in light_models.split(",")
            if item.strip()
        ]

        settings.heavy_models = [
            item.strip()
            for item in heavy_models.split(",")
            if item.strip()
        ]

        settings.clip_model_id = (
            clip_model.strip() or settings.clip_model_id
        )

    st.divider()

    if analysis_mode == "Deep jury":
        st.caption(
            "Privacy: ad text and up to four resized photos are sent "
            "through OpenRouter to the selected providers."
        )
    else:
        st.caption(
            "Private mode: images stay local. Persian semantic matching "
            "is less accurate without the light translator."
        )

    st.caption(
        "Scores are decision support—not fraud detection, appraisal, "
        "or a publishing guarantee."
    )


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">AI LISTING INTELLIGENCE</div>
      <h1>Will buyers believe what they see?</h1>
      <p>
        Upload a Divar-style ad. CLIP-ViT checks whether every photo
        belongs to the story; a routed AI jury checks clarity, evidence,
        and trust.
      </p>
      <div class="micro-row">
        <span class="micro">CLIP semantic match</span>
        <span class="micro">outlier photos</span>
        <span class="micro">duplicate angles</span>
        <span class="micro">copy quality</span>
        <span class="micro">seller context</span>
        <span class="micro">multi-LLM jury</span>
        <span class="micro">0–100 explainable score</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


sample_col, note_col = st.columns([1, 3])

with sample_col:
    if st.button("Load Persian sample", use_container_width=True):
        st.session_state["ad_title"] = (
            "مبل راحتی هفت نفره طوسی، سالم و تمیز"
        )

        st.session_state["ad_description"] = (
            "مبل هفت نفره شامل یک کاناپه سه نفره و دو مبل تک‌نفره است. "
            "سه سال استفاده شده، پارچه بدون پارگی و کلاف کاملاً سالم است. "
            "روی دسته سمت چپ یک لکه کوچک دارد که در عکس نزدیک مشخص شده. "
            "به علت جابه‌جایی فروخته می‌شود. بازدید در محل امکان‌پذیر است."
        )

        st.session_state["ad_price"] = "۳۸٬۰۰۰٬۰۰۰ تومان"

with note_col:
    st.caption(
        "The sample fills the copy only—add your own photos "
        "so the visual evidence is real."
    )


with st.form("ad_form", clear_on_submit=False):
    left, right = st.columns(
        [1.05, 0.95],
        gap="large",
    )

    with left:
        st.markdown("### 1 · Tell the listing story")

        title = st.text_input(
            "عنوان آگهی / Ad title",
            key="ad_title",
            placeholder="مثلاً: آیفون ۱۳، ۱۲۸ گیگ، بدون تعمیر",
        )

        description = st.text_area(
            "توضیحات / Description",
            key="ad_description",
            height=210,
            placeholder=(
                "Condition, history, included items, visible defects, "
                "and reason for selling…"
            ),
        )

        field_a, field_b = st.columns(2)

        with field_a:
            category = st.selectbox(
                "Category",
                list(CATEGORY_VISUAL_HINTS),
            )

            city = st.text_input(
                "City / area",
                placeholder="Tehran, Saadat Abad",
            )

            condition = st.selectbox(
                "وضعیت کالا",
                [
                    "نو",
                    "در حد نو",
                    "سالم",
                    "استفاده‌شده",
                    "نیازمند تعمیر",
                    "قطعاتی",
                    "نامشخص",
                ],
                index=2,
            )

            ownership_status = st.selectbox(
                "وضعیت مالکیت",
                [
                    "مالک اول",
                    "مالک دوم یا بیشتر",
                    "وکالتی",
                    "شرکتی",
                    "نامشخص",
                ],
            )

            seller_type = st.selectbox(
                "نوع فروشنده",
                [
                    "شخصی",
                    "فروشگاه",
                    "تولیدی / کسب‌وکار",
                    "نمایندگی",
                    "نامشخص",
                ],
            )

        with field_b:
            price = st.text_input(
                "Price",
                key="ad_price",
                placeholder="مثلاً ۲۵٬۰۰۰٬۰۰۰ تومان",
            )

            usage_duration = st.text_input(
                "مدت استفاده",
                placeholder="مثلاً ۲ سال",
            )

            manufacture_year = st.number_input(
                "سال تولید",
                min_value=1900,
                max_value=2100,
                value=None,
                step=1,
                format="%d",
            )

            negotiable = st.checkbox(
                "قیمت قابل مذاکره است",
            )

            exchange_possible = st.checkbox(
                "امکان معاوضه وجود دارد",
            )

        st.markdown("#### جزئیات اعتماد و شفافیت")

        defects = st.text_area(
            "ایرادها، خط‌وخش یا تعمیرات",
            height=100,
            placeholder=(
                "تمام ایرادهای ظاهری و فنی، تعمیرات، تعویض قطعات "
                "یا مواردی که خریدار باید بداند…"
            ),
        )

        included_items = st.text_input(
            "اقلام همراه",
            placeholder="جعبه، شارژر، ریموت، سند، لوازم جانبی و…",
        )

        reason_for_selling = st.text_input(
            "دلیل فروش",
            placeholder="جابجایی، ارتقا، عدم نیاز و…",
        )

        seller_note = st.text_area(
            "یادداشت فروشنده",
            height=90,
            placeholder="شرایط پرداخت، محدودیت زمانی یا توضیحات تکمیلی…",
        )

    with right:
        st.markdown("### 2 · Add the visual evidence")

        uploaded = st.file_uploader(
            "Upload 1–10 photos",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help=(
                "For the best score, show multiple useful angles "
                "rather than repeated frames."
            ),
        )

        if uploaded:
            preview_columns = st.columns(min(3, len(uploaded)))

            for index, file in enumerate(uploaded[:6]):
                with preview_columns[index % len(preview_columns)]:
                    st.image(file, use_container_width=True)
                    st.caption(file.name)
        else:
            st.info(
                "Add real listing photos. One image works, but 3–6 distinct "
                "angles produce stronger evidence."
            )

        st.markdown("#### معامله و خدمات")

        transaction_method = st.multiselect(
            "روش‌های معامله",
            [
                "حضوری",
                "پرداخت در محل",
                "کارت‌به‌کارت",
                "درگاه امن",
                "ارسال با پست",
                "پیک",
            ],
            default=["حضوری"],
        )

        delivery_options = st.multiselect(
            "امکانات تحویل",
            [
                "تحویل حضوری",
                "ارسال داخل شهر",
                "ارسال به شهرستان",
                "هماهنگی با خریدار",
            ],
        )

        viewing_available = st.checkbox(
            "امکان بازدید یا تست وجود دارد",
        )

        original_receipt_available = st.checkbox(
            "فاکتور یا رسید خرید موجود است",
        )

        warranty_available = st.checkbox(
            "کالا دارای ضمانت یا گارانتی است",
        )

        st.markdown("#### تأیید اطلاعات")

        information_confirmed = st.checkbox(
            "اطلاعات واردشده تا حد امکان دقیق و کامل است",
        )

    analyze_clicked = st.form_submit_button(
        "Score this ad",
        type="primary",
        use_container_width=True,
    )


if analyze_clicked:
    if not title.strip() or not description.strip():
        st.error("Please provide both a title and description.")

    elif not uploaded:
        st.error(
            "Please upload at least one image; this scorer is intentionally multimodal."
        )

    elif not price.strip():
        st.error("Please provide a price.")

    elif not information_confirmed:
        st.error(
            "لطفاً صحت و کامل‌بودن اطلاعات واردشده را تأیید کنید."
        )

    elif (
        analysis_mode == "Deep jury"
        and not settings.openrouter_api_key
    ):
        st.error(
            "Deep jury needs an OpenRouter API key. "
            "Add one in the sidebar or choose Local-only preview."
        )

    else:
        images, names, image_errors = _load_images(uploaded)

        if image_errors:
            st.warning(
                "Some files could not be read: "
                + " | ".join(image_errors)
            )

        if not images:
            st.error("None of the uploaded files could be decoded as images.")

        else:
            ad = AdInput(
                title=title.strip(),
                description=description.strip(),
                category=category,
                city=city.strip(),
                price=price.strip(),
                images=images,
                image_names=names,

                condition=condition,
                usage_duration=usage_duration.strip(),
                manufacture_year=manufacture_year,
                ownership_status=ownership_status,
                seller_type=seller_type,

                defects=defects.strip(),
                included_items=included_items.strip(),
                reason_for_selling=reason_for_selling.strip(),
                seller_note=seller_note.strip(),

                transaction_method=transaction_method,
                delivery_options=delivery_options,

                negotiable=negotiable,
                exchange_possible=exchange_possible,
                viewing_available=viewing_available,
                original_receipt_available=(
                    original_receipt_available
                ),
                warranty_available=warranty_available,
            )

            try:
                with st.status(
                    "Building the evidence graph…",
                    expanded=True,
                ) as status:
                    st.write(
                        "Running deterministic copy and image-quality checks"
                    )

                    local_text = analyze_text(ad)
                    image_signals = analyze_image_quality(images)

                    jury = OpenRouterJury(settings)

                    if analysis_mode == "Deep jury":
                        st.write(
                            "Dispatching two light specialists in parallel"
                        )
                        llm_review = jury.light_review(ad)
                    else:
                        llm_review = LLMReview(enabled=False)

                    visual_caption = str(
                        llm_review.visual_brief.get(
                            "visual_caption",
                            "",
                        )
                    ).strip()

                    visual_caption = (
                        visual_caption
                        or _fallback_visual_caption(ad)
                    )

                    st.write(
                        f"Running CLIP-ViT on {len(images)} photo(s)"
                    )

                    clip_signals = _clip_analyzer(
                        settings.clip_model_id
                    ).analyze(
                        images,
                        visual_caption,
                    )

                    if analysis_mode == "Deep jury":
                        st.write(
                            "Sending compact evidence to the "
                            "heavy multimodal judge"
                        )

                        llm_review = jury.heavy_review(
                            ad,
                            clip_signals,
                            image_signals,
                            local_text,
                            llm_review,
                        )

                    st.write(
                        "Composing the transparent weighted score"
                    )

                    result = ScoreComposer().compose(
                        clip_signals,
                        image_signals,
                        local_text,
                        llm_review,
                        len(images),
                    )

                    if not hasattr(result, "metadata"):
                        result.metadata = {}

                    result.metadata["ad_context"] = (
                        _build_ad_context(ad)
                    )

                    result.metadata["text_analysis"] = (
                        local_text.metadata
                    )

                    status.update(
                        label="Evidence review complete",
                        state="complete",
                        expanded=False,
                    )

                st.session_state["analysis_result"] = result
                st.session_state["analysis_images"] = images
                st.session_state["analysis_names"] = names
                st.session_state["clip_model_used"] = (
                    settings.clip_model_id
                )

            except Exception as exc:
                st.exception(exc)
                st.info(
                    "If this is the first run, confirm internet access "
                    "so Transformers can download the CLIP checkpoint. "
                    "The model is cached locally after that."
                )


if "analysis_result" in st.session_state:
    st.divider()

    _render_result(
        st.session_state["analysis_result"],
        st.session_state.get("analysis_images", []),
        st.session_state.get("analysis_names", []),
    )


st.markdown(
    """
    <div class="section-card" style="margin-top:30px">
      <strong>How to read the score</strong><br>
      <span class="tiny">
        85–100 publish-ready · 70–84 strong · 50–69 needs focused edits
        · below 50 major evidence or trust gaps.
        Scores compare the ad to good marketplace practice;
        they do not estimate the item's price.
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)