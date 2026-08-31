"""Local-first voice NSFW detector with keyword spotting.
Audio is transcribed locally with faster-whisper.

Only transcript chunks that cross the local suspicion threshold are optionally
sent to OpenRouter for semantic verification. Raw audio is never uploaded to
OpenRouter by this app.

Keyword spotting uses faster-whisper word-level timestamps. Keyword hits are
shown with their approximate start/end times.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests
import streamlit as st


APP_TITLE = "Voice NSFW Detector"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_KEYWORD_MIN_PROBABILITY = 0.35
DEFAULT_FUZZY_RATIO = 0.90


# Keep this list deliberately small and editable.
# The local stage is a triage filter, not a complete moderation model.
DEFAULT_TERM_WEIGHTS: dict[str, float] = {
    # Explicit sexual language and anatomy.
    "anal": 2.5,
    "blowjob": 3.0,
    "breast": 1.8,
    "clitoris": 3.0,
    "cock": 2.5,
    "cum": 2.5,
    "dick": 2.5,
    "dildo": 2.5,
    "ejaculate": 3.0,
    "erection": 2.0,
    "fingering": 3.0,
    "fuck": 2.2,
    "handjob": 3.0,
    "masturbat": 3.0,
    "moan": 1.8,
    "naked": 1.5,
    "nipple": 2.0,
    "nude": 1.5,
    "orgasm": 2.5,
    "penis": 2.5,
    "porn": 2.5,
    "pussy": 2.5,
    "sex": 1.8,
    "semen": 2.5,
    "sexual": 1.8,
    "sperm": 2.5,
    "suck": 1.3,
    "tits": 2.0,
    "vagina": 2.5,
    "vibrator": 2.5,
    "vulva": 2.5,
    # Lower-weight profanity by itself.
    "ass": 0.8,
    "bastard": 0.7,
    "bitch": 0.8,
    "bullshit": 0.7,
    "shit": 0.7,
}


# Regex phrase signals used by the fallback text scorer.
PHRASE_WEIGHTS: dict[str, float] = {
    r"\b(come|cum)\s+on\s+(me|your|her|him)\b": 3.0,
    r"\b(send|show)\s+(me\s+)?(nudes?|a\s+nude)\b": 3.0,
    r"\b(want|wanna)\s+have\s+sex\b": 3.0,
    r"\b(very\s+)?explicit\s+content\b": 2.0,
}


# Literal multi-word phrases used by the word-level spotter.
DEFAULT_KEYWORD_PHRASES: dict[str, float] = {
    "have sex": 3.0,
    "want sex": 3.0,
    "send nudes": 3.5,
    "show me your body": 3.0,
    "explicit content": 2.0,
}


TERM_ALIASES: dict[str, str] = {
    "f*ck": "fuck",
    "f**k": "fuck",
    "fuk": "fuck",
    "phuck": "fuck",
    "sh1t": "shit",
    "b1tch": "bitch",
    "a55": "ass",
    "d1ck": "dick",
    "p0rn": "porn",
}


@dataclass
class Spot:
    """A single keyword or phrase match with timing information."""

    keyword: str
    matched_text: str
    start: float
    end: float
    probability: float | None = None
    fuzzy: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptChunk:
    index: int
    start: float
    end: float
    text: str
    local_score: float = 0.0
    local_matches: list[str] = field(default_factory=list)
    keyword_spots: list[Spot] = field(default_factory=list)
    locally_suspicious: bool = False
    openrouter_reviewed: bool = False
    openrouter_result: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "local_score": self.local_score,
            "local_matches": self.local_matches,
            "keyword_spots": [
                spot.as_dict() for spot in self.keyword_spots
            ],
            "locally_suspicious": self.locally_suspicious,
            "openrouter_reviewed": self.openrouter_reviewed,
            "openrouter_result": self.openrouter_result,
        }


def normalize_text(text: str) -> str:
    """Normalize casing, punctuation, aliases, and light leetspeak."""

    normalized = str(text or "").lower().replace("’", "'")

    normalized = re.sub(
        r"[^\w\s*'-]",
        " ",
        normalized,
        flags=re.UNICODE,
    )

    for alias, replacement in TERM_ALIASES.items():
        normalized = normalized.replace(alias, replacement)

    # Conservative leetspeak normalization.
    normalized = re.sub(r"(?<=[a-z])0(?=[a-z])", "o", normalized)
    normalized = re.sub(r"(?<=[a-z])1(?=[a-z])", "i", normalized)
    normalized = re.sub(r"(?<=[a-z])3(?=[a-z])", "e", normalized)

    return re.sub(r"\s+", " ", normalized).strip()


def normalize_token(text: str) -> str:
    """Normalize an individual ASR word."""

    return normalize_text(text).strip("'-_*")


def _term_matches(text: str, terms: Iterable[str]) -> list[str]:
    """Match configured terms against normalized chunk text."""

    matches: list[str] = []
    tokens = text.split()

    for term in terms:
        term = normalize_token(term)

        if not term:
            continue

        # Stem-like entries such as "masturbat" match masturbate/masturbation.
        pattern = rf"\b{re.escape(term)}\w*\b"

        if re.search(pattern, text):
            matches.append(term)
            continue

        # Fuzzy fallback for longer words.
        if len(term) >= 5 and any(
            difflib.SequenceMatcher(None, token, term).ratio()
            >= DEFAULT_FUZZY_RATIO
            for token in tokens
            if len(token) >= 4
        ):
            matches.append(f"~{term}")

    return matches


def score_text(
    text: str,
    term_weights: dict[str, float],
) -> tuple[float, list[str]]:
    """Return the chunk-level fallback suspicion score."""

    normalized = normalize_text(text)

    if not normalized:
        return 0.0, []

    score = 0.0
    matches = _term_matches(normalized, term_weights)

    for match in matches:
        score += term_weights.get(match.lstrip("~"), 0.0)

    for pattern, weight in PHRASE_WEIGHTS.items():
        if re.search(pattern, normalized):
            score += weight
            matches.append("phrase")

    # Multiple signals in one chunk are more meaningful than one ambiguous term.
    if len(matches) >= 2:
        score += min(2.0, 0.5 * (len(matches) - 1))

    return round(score, 2), sorted(set(matches))


def _word_probability(word: Any) -> float | None:
    """Safely read faster-whisper's word probability."""

    value = getattr(word, "probability", None)

    if value is None:
        return None

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None

    if not 0.0 <= probability <= 1.0:
        return None

    return probability


def _word_is_usable(
    word: Any,
    minimum_probability: float,
) -> bool:
    """Return whether a word is reliable enough for keyword spotting."""

    probability = _word_probability(word)

    # Some faster-whisper versions or configurations may not expose a
    # probability. In that case, keep the word rather than losing recall.
    return (
        probability is None
        or probability >= minimum_probability
    )


def _fuzzy_word_match(
    token: str,
    term: str,
    enabled: bool,
    fuzzy_ratio: float,
) -> bool:
    """Compare a token with a keyword using conservative fuzzy matching."""

    if not enabled:
        return False

    if len(term) < 5 or len(token) < 4:
        return False

    return (
        difflib.SequenceMatcher(None, token, term).ratio()
        >= fuzzy_ratio
    )


def _keyword_matches_token(
    token: str,
    keyword: str,
    fuzzy_enabled: bool,
    fuzzy_ratio: float,
) -> tuple[bool, bool]:
    """
    Return (matched, fuzzy).

    Stem-like terms such as "masturbat" match "masturbation".
    """

    token = normalize_token(token)
    keyword = normalize_token(keyword)

    if not token or not keyword:
        return False, False

    if token == keyword or token.startswith(keyword):
        return True, False

    if _fuzzy_word_match(
        token,
        keyword,
        enabled=fuzzy_enabled,
        fuzzy_ratio=fuzzy_ratio,
    ):
        return True, True

    return False, False


def _segment_words(segment: Any) -> list[Any]:
    """Return word-level results if available."""

    words = getattr(segment, "words", None) or []
    return list(words)


def spot_keywords(
    segments: Iterable[Any],
    term_weights: dict[str, float],
    phrase_weights: dict[str, float],
    minimum_probability: float = DEFAULT_KEYWORD_MIN_PROBABILITY,
    fuzzy_enabled: bool = True,
    fuzzy_ratio: float = DEFAULT_FUZZY_RATIO,
) -> list[Spot]:
    """
    Find single-word and multi-word keyword hits.

    This requires transcription with word_timestamps=True.
    """

    terms = {
        normalize_token(term): weight
        for term, weight in term_weights.items()
        if normalize_token(term)
    }

    phrases: dict[tuple[str, ...], tuple[str, float]] = {}

    for phrase, weight in phrase_weights.items():
        normalized_phrase = normalize_text(phrase)
        phrase_tokens = tuple(normalized_phrase.split())

        if phrase_tokens:
            phrases[phrase_tokens] = (
                normalized_phrase,
                weight,
            )

    spots: list[Spot] = []

    for segment in segments:
        words = [
            word
            for word in _segment_words(segment)
            if str(getattr(word, "word", "")).strip()
            and _word_is_usable(word, minimum_probability)
        ]

        if not words:
            continue

        normalized_words = [
            normalize_token(
                str(getattr(word, "word", ""))
            )
            for word in words
        ]

        # Single-word keyword spotting.
        for word, token in zip(words, normalized_words):
            for keyword in terms:
                matched, fuzzy = _keyword_matches_token(
                    token=token,
                    keyword=keyword,
                    fuzzy_enabled=fuzzy_enabled,
                    fuzzy_ratio=fuzzy_ratio,
                )

                if not matched:
                    continue

                segment_start = float(
                    getattr(segment, "start", 0.0) or 0.0
                )
                segment_end = float(
                    getattr(segment, "end", segment_start)
                    or segment_start
                )

                start = float(
                    getattr(word, "start", segment_start)
                    or segment_start
                )
                end = float(
                    getattr(word, "end", segment_end)
                    or start
                )

                spots.append(
                    Spot(
                        keyword=keyword,
                        matched_text=str(
                            getattr(word, "word", keyword)
                        ).strip(),
                        start=start,
                        end=max(start, end),
                        probability=_word_probability(word),
                        fuzzy=fuzzy,
                    )
                )

                # Do not match the same ASR word against additional terms.
                break

        # Multi-word phrase spotting.
        for phrase_tokens, (
            display_phrase,
            _,
        ) in phrases.items():
            phrase_length = len(phrase_tokens)

            if phrase_length == 0:
                continue

            for index in range(
                len(normalized_words) - phrase_length + 1
            ):
                window = normalized_words[
                    index : index + phrase_length
                ]

                if window != list(phrase_tokens):
                    continue

                first_word = words[index]
                last_word = words[index + phrase_length - 1]

                segment_start = float(
                    getattr(segment, "start", 0.0) or 0.0
                )
                segment_end = float(
                    getattr(segment, "end", segment_start)
                    or segment_start
                )

                start = float(
                    getattr(first_word, "start", segment_start)
                    or segment_start
                )
                end = float(
                    getattr(last_word, "end", segment_end)
                    or start
                )

                probabilities = [
                    _word_probability(words[position])
                    for position in range(
                        index,
                        index + phrase_length,
                    )
                ]

                known_probabilities = [
                    value
                    for value in probabilities
                    if value is not None
                ]

                matched_text = " ".join(
                    str(getattr(item, "word", "")).strip()
                    for item in words[
                        index : index + phrase_length
                    ]
                )

                spots.append(
                    Spot(
                        keyword=display_phrase,
                        matched_text=matched_text,
                        start=start,
                        end=max(start, end),
                        probability=(
                            min(known_probabilities)
                            if known_probabilities
                            else None
                        ),
                        fuzzy=False,
                    )
                )

    # Remove exact duplicates.
    unique: dict[tuple[str, float, float], Spot] = {}

    for spot in spots:
        key = (
            spot.keyword,
            round(spot.start, 2),
            round(spot.end, 2),
        )
        unique[key] = spot

    return sorted(
        unique.values(),
        key=lambda item: (
            item.start,
            item.end,
            item.keyword,
        ),
    )


def score_keyword_spots(
    spots: Iterable[Spot],
    term_weights: dict[str, float],
    phrase_weights: dict[str, float],
) -> tuple[float, list[str]]:
    """Convert keyword spots into a local suspicion score."""

    score = 0.0
    matches: list[str] = []
    seen_signal_keys: set[tuple[str, int]] = set()

    for spot in spots:
        time_bucket = int(spot.start // 1)
        signal_key = (spot.keyword, time_bucket)

        # Prevent repeated identical detections in the same second from
        # inflating the score.
        if signal_key in seen_signal_keys:
            continue

        seen_signal_keys.add(signal_key)

        weight = phrase_weights.get(
            spot.keyword,
            term_weights.get(spot.keyword, 0.0),
        )

        # Fuzzy matches are discounted because they may be ASR errors.
        if spot.fuzzy:
            weight *= 0.85

        score += weight

        label = spot.keyword
        if spot.fuzzy:
            label = f"~{label}"

        matches.append(
            f"{label}@{_seconds(spot.start)}"
        )

    unique_signal_count = len(seen_signal_keys)

    if unique_signal_count >= 2:
        score += min(
            2.0,
            0.5 * (unique_signal_count - 1),
        )

    return round(score, 2), sorted(set(matches))


def _seconds(value: float) -> str:
    minutes, seconds = divmod(max(0.0, value), 60.0)
    return f"{int(minutes):02d}:{seconds:04.1f}"


def build_chunks(
    segments: Iterable[Any],
    chunk_seconds: int,
    duration: float | None = None,
) -> list[TranscriptChunk]:
    """Group local ASR segments into fixed time windows."""

    buckets: dict[int, list[Any]] = {}

    for segment in segments:
        text = str(getattr(segment, "text", "")).strip()

        if not text:
            continue

        start = float(
            getattr(segment, "start", 0.0) or 0.0
        )
        bucket = int(start // chunk_seconds)
        buckets.setdefault(bucket, []).append(segment)

    chunks: list[TranscriptChunk] = []

    for index, bucket in enumerate(sorted(buckets)):
        bucket_segments = buckets[bucket]

        start = min(
            float(getattr(item, "start", 0.0) or 0.0)
            for item in bucket_segments
        )

        end = max(
            float(
                getattr(item, "end", start)
                or start
            )
            for item in bucket_segments
        )

        if duration:
            end = min(end, duration)

        text = " ".join(
            str(getattr(item, "text", "")).strip()
            for item in bucket_segments
        )

        chunks.append(
            TranscriptChunk(
                index=index,
                start=start,
                end=max(start, end),
                text=text,
            )
        )

    return chunks


@st.cache_resource(show_spinner=False)
def load_whisper_model(
    model_size: str,
    device: str,
    compute_type: str,
) -> Any:
    """Load and cache the local ASR model."""

    from faster_whisper import WhisperModel

    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )


def transcribe_audio(
    audio_path: str,
    model_size: str,
    device: str,
    compute_type: str,
    language: str,
) -> tuple[list[Any], float, str]:
    """Transcribe audio locally with word-level timestamps."""

    model = load_whisper_model(
        model_size,
        device,
        compute_type,
    )

    segments, info = model.transcribe(
        audio_path,
        language=None if language == "auto" else language,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 300,
        },
        condition_on_previous_text=False,
        word_timestamps=True,
    )

    segment_list = list(segments)

    duration = float(
        getattr(info, "duration", 0.0) or 0.0
    )

    if duration <= 0 and segment_list:
        duration = max(
            float(
                getattr(item, "end", 0.0) or 0.0
            )
            for item in segment_list
        )

    detected_language = str(
        getattr(info, "language", "unknown")
        or "unknown"
    )

    return (
        segment_list,
        duration,
        detected_language,
    )


def _openrouter_key() -> str:
    """Read the OpenRouter API key from Streamlit secrets or environment."""

    try:
        secret_value = st.secrets.get(
            "OPENROUTER_API_KEY",
            "",
        )
    except Exception:
        secret_value = ""

    return str(
        secret_value
        or os.getenv("OPENROUTER_API_KEY", "")
    ).strip()


def _extract_json(content: Any) -> dict[str, Any]:
    """Extract an object from a JSON or fenced JSON response."""

    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    raw = str(content or "").strip()

    raw = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    match = re.search(
        r"\{.*\}",
        raw,
        flags=re.DOTALL,
    )

    if match:
        raw = match.group(0)

    parsed = json.loads(raw)

    if not isinstance(parsed, dict):
        raise ValueError(
            "OpenRouter returned a non-object JSON value"
        )

    return parsed


def review_with_openrouter(
    text: str,
    model: str,
    api_key: str,
    referer: str = "",
    app_name: str = APP_TITLE,
) -> dict[str, Any]:
    """Review transcript text only; never send uploaded audio bytes."""

    system_prompt = (
        "You are a strict content-safety classifier. Classify the supplied "
        "speech transcript for adult/sexually explicit NSFW content. Do not "
        "quote or expand the content. Return only JSON with these keys: "
        "is_nsfw (boolean), confidence (number 0 to 1), categories "
        "(array of short strings), reason (short string). Profanity without "
        "sexual meaning is not enough by itself. Consider ASR errors and "
        "context."
    )

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Transcript chunk:\n{text}",
            },
        ],
        "temperature": 0,
        "max_tokens": 180,
        "response_format": {
            "type": "json_object",
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": app_name,
    }

    if referer.strip():
        headers["HTTP-Referer"] = referer.strip()

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=90,
    )

    # Some models do not support response_format. Retry without it.
    if (
        response.status_code == 400
        and "response_format" in payload
    ):
        payload.pop("response_format")

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=90,
        )

    response.raise_for_status()

    body = response.json()
    content = body["choices"][0]["message"]["content"]
    result = _extract_json(content)

    result["is_nsfw"] = bool(
        result.get("is_nsfw", False)
    )

    try:
        result["confidence"] = max(
            0.0,
            min(
                1.0,
                float(result.get("confidence", 0)),
            ),
        )
    except (TypeError, ValueError):
        result["confidence"] = 0.0

    if not isinstance(
        result.get("categories"),
        list,
    ):
        result["categories"] = []

    result["reason"] = str(
        result.get("reason", "")
    ).strip()

    return result


def openrouter_label(chunk: TranscriptChunk) -> str:
    """Return a readable OpenRouter status."""

    if not chunk.openrouter_reviewed:
        return "Not sent"

    review = chunk.openrouter_result or {}

    if "error" in (
        review.get("categories") or []
    ):
        return "Review error"

    return (
        "NSFW"
        if review.get("is_nsfw")
        else "Not NSFW"
    )


def render_results(
    chunks: list[TranscriptChunk],
) -> None:
    """Render chunk-level results."""

    rows: list[dict[str, Any]] = []

    for chunk in chunks:
        keyword_spots = "; ".join(
            (
                f"{spot.keyword} "
                f"({_seconds(spot.start)}–"
                f"{_seconds(spot.end)})"
                + (" [fuzzy]" if spot.fuzzy else "")
            )
            for spot in chunk.keyword_spots
        )

        rows.append(
            {
                "Chunk": chunk.index + 1,
                "Time": (
                    f"{_seconds(chunk.start)}–"
                    f"{_seconds(chunk.end)}"
                ),
                "Local score": chunk.local_score,
                "Local matches": ", ".join(
                    chunk.local_matches
                ),
                "Keyword spots": keyword_spots,
                "OpenRouter": openrouter_label(chunk),
                "Transcript": chunk.text,
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )


def render_keyword_hits(
    chunks: list[TranscriptChunk],
) -> None:
    """Render word-level keyword hits."""

    spots = [
        spot
        for chunk in chunks
        for spot in chunk.keyword_spots
    ]

    if not spots:
        st.info("No configured keywords were spotted.")
        return

    rows = [
        {
            "Keyword": spot.keyword,
            "Recognized text": spot.matched_text,
            "Time": (
                f"{_seconds(spot.start)}–"
                f"{_seconds(spot.end)}"
            ),
            "Probability": (
                round(spot.probability, 3)
                if spot.probability is not None
                else None
            ),
            "Fuzzy match": spot.fuzzy,
        }
        for spot in spots
    ]

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🔎",
        layout="wide",
    )

    st.title("🔎 Voice NSFW Detector")

    st.caption(
        "Local-first moderation: audio is transcribed locally, and only "
        "suspicious transcript chunks are optionally sent to OpenRouter."
    )

    with st.sidebar:
        st.header("Detection settings")

        model_size = st.selectbox(
            "Local Whisper model",
            [
                "tiny",
                "base",
                "small",
                "medium",
            ],
            index=1,
            help=(
                "Larger models are usually more accurate but require "
                "more RAM and processing time."
            ),
        )

        device = st.selectbox(
            "Device",
            ["cpu", "cuda"],
            index=0,
        )

        compute_type = st.selectbox(
            "Compute type",
            [
                "int8",
                "float16",
                "float32",
            ],
            index=0 if device == "cpu" else 1,
        )

        language = st.selectbox(
            "Spoken language",
            [
                "auto",
                "en",
                "fa",
                "de",
                "es",
                "fr",
                "ar",
            ],
            index=0,
        )

        chunk_seconds = st.slider(
            "Transcript chunk size (seconds)",
            5,
            30,
            12,
        )

        local_threshold = st.slider(
            "Local suspicion threshold",
            0.5,
            8.0,
            2.0,
            0.5,
        )

        keyword_min_probability = st.slider(
            "Minimum keyword confidence",
            0.0,
            1.0,
            DEFAULT_KEYWORD_MIN_PROBABILITY,
            0.05,
            help=(
                "Ignore keyword hits when faster-whisper gives the "
                "word a lower probability. If probabilities are not "
                "available, the word is retained."
            ),
        )

        enable_fuzzy_keywords = st.checkbox(
            "Enable fuzzy keyword matching",
            value=True,
            help=(
                "Improves recall for ASR spelling errors but can "
                "increase false positives."
            ),
        )

        fuzzy_ratio = st.slider(
            "Fuzzy match ratio",
            0.80,
            1.00,
            DEFAULT_FUZZY_RATIO,
            0.01,
            disabled=not enable_fuzzy_keywords,
        )

        max_reviews = st.slider(
            "Maximum OpenRouter reviews",
            0,
            30,
            10,
        )

        st.divider()
        st.header("Custom keywords")

        custom_terms = st.text_area(
            "Extra local keywords",
            placeholder=(
                "Comma-separated, for example: "
                "keyword1, keyword2"
            ),
            help=(
                "Each custom single-word keyword receives a default "
                "local weight of 2.0."
            ),
        )

        custom_phrases = st.text_area(
            "Extra keyword phrases",
            placeholder=(
                "One phrase per line, for example:\n"
                "private video\n"
                "send me pictures"
            ),
            help=(
                "Each custom phrase receives a default local weight "
                "of 3.0."
            ),
        )

        st.divider()
        st.header("OpenRouter (optional)")

        openrouter_model = st.text_input(
            "Model",
            value=os.getenv(
                "OPENROUTER_MODEL",
                "openai/gpt-4o-mini",
            ),
        )

        openrouter_key = _openrouter_key()

        if openrouter_key:
            st.success("API key loaded")
        else:
            st.info("No API key: local-only mode")

        referer = st.text_input(
            "HTTP-Referer (optional)",
            value=os.getenv(
                "OPENROUTER_REFERER",
                "",
            ),
        )

    uploaded = st.file_uploader(
        "Upload an audio file",
        type=[
            "wav",
            "mp3",
            "m4a",
            "flac",
            "ogg",
            "webm",
        ],
        accept_multiple_files=False,
    )

    if not uploaded:
        st.info(
            "Upload a voice recording to begin. The first run may download "
            "the local Whisper model."
        )
        return

    audio_bytes = uploaded.getvalue()

    st.audio(
        audio_bytes,
        format=uploaded.type or "audio/wav",
    )

    analyze = st.button(
        "Analyze voice",
        type="primary",
        use_container_width=True,
    )

    if not analyze:
        return

    # Build the custom single-word term dictionary.
    terms = dict(DEFAULT_TERM_WEIGHTS)

    for raw_term in custom_terms.split(","):
        normalized_term = normalize_token(raw_term)

        if normalized_term:
            terms[normalized_term] = max(
                terms.get(normalized_term, 0.0),
                2.0,
            )

    # Build literal phrase dictionary.
    phrase_weights = dict(DEFAULT_KEYWORD_PHRASES)

    for raw_phrase in custom_phrases.splitlines():
        normalized_phrase = normalize_text(raw_phrase)

        if normalized_phrase:
            phrase_weights[normalized_phrase] = max(
                phrase_weights.get(normalized_phrase, 0.0),
                3.0,
            )

    suffix = Path(uploaded.name).suffix or ".audio"
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as handle:
            handle.write(audio_bytes)
            temp_path = handle.name

        with st.status(
            "Transcribing locally…",
            expanded=True,
        ) as status:
            segments, duration, detected_language = (
                transcribe_audio(
                    temp_path,
                    model_size,
                    device,
                    compute_type,
                    language,
                )
            )

            # Keep the list because faster-whisper returns a generator.
            keyword_spots = spot_keywords(
                segments=segments,
                term_weights=terms,
                phrase_weights=phrase_weights,
                minimum_probability=keyword_min_probability,
                fuzzy_enabled=enable_fuzzy_keywords,
                fuzzy_ratio=fuzzy_ratio,
            )

            chunks = build_chunks(
                segments,
                chunk_seconds,
                duration,
            )

            for chunk in chunks:
                chunk.keyword_spots = [
                    spot
                    for spot in keyword_spots
                    if (
                        spot.start < chunk.end
                        and spot.end >= chunk.start
                    )
                ]

                keyword_score, keyword_matches = (
                    score_keyword_spots(
                        chunk.keyword_spots,
                        terms,
                        phrase_weights,
                    )
                )

                text_score, text_matches = score_text(
                    chunk.text,
                    terms,
                )

                if chunk.keyword_spots:
                    # Avoid double-counting the same signal by taking the
                    # stronger of the word-level and fallback scores.
                    chunk.local_score = max(
                        keyword_score,
                        text_score,
                    )
                    chunk.local_matches = sorted(
                        set(
                            keyword_matches
                            + text_matches
                        )
                    )
                else:
                    chunk.local_score = text_score
                    chunk.local_matches = text_matches

                chunk.locally_suspicious = (
                    chunk.local_score >= local_threshold
                )

            status.update(
                label="Local transcription complete",
                state="complete",
            )

    except Exception as exc:
        st.error(
            "Local transcription failed. Check that faster-whisper is "
            "installed, the selected device/compute type is supported, and "
            f"the audio format is readable. Details: {exc}"
        )
        return

    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    suspicious = [
        chunk
        for chunk in chunks
        if chunk.locally_suspicious
    ]

    review_count = min(
        len(suspicious),
        max_reviews,
    )

    if review_count and openrouter_key:
        progress = st.progress(
            0,
            text="Reviewing suspicious transcript chunks…",
        )

        for position, chunk in enumerate(
            suspicious[:review_count],
            start=1,
        ):
            try:
                chunk.openrouter_result = (
                    review_with_openrouter(
                        text=chunk.text,
                        model=openrouter_model.strip(),
                        api_key=openrouter_key,
                        referer=referer,
                    )
                )
                chunk.openrouter_reviewed = True

            except Exception as exc:
                chunk.openrouter_result = {
                    "is_nsfw": False,
                    "confidence": 0.0,
                    "categories": ["error"],
                    "reason": (
                        "OpenRouter review failed: "
                        f"{exc}"
                    ),
                }
                chunk.openrouter_reviewed = True

            progress.progress(
                position / review_count,
                text=(
                    f"Reviewed {position}/"
                    f"{review_count}"
                ),
            )

        progress.empty()

    local_only_flagged = sum(
        chunk.locally_suspicious
        for chunk in chunks
    )

    if review_count and openrouter_key:
        # Preserve local flags for unreviewed chunks and API failures.
        final_flagged = sum(
            chunk.locally_suspicious
            and (
                not chunk.openrouter_reviewed
                or not chunk.openrouter_result
                or "error" in (
                    chunk.openrouter_result.get(
                        "categories",
                        [],
                    )
                )
                or bool(
                    chunk.openrouter_result.get(
                        "is_nsfw",
                        False,
                    )
                )
            )
            for chunk in chunks
        )
    else:
        final_flagged = local_only_flagged

    st.subheader("Summary")

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Detected language",
        detected_language,
    )

    metric_columns[1].metric(
        "Duration",
        _seconds(duration),
    )

    metric_columns[2].metric(
        "Transcript chunks",
        len(chunks),
    )

    metric_columns[3].metric(
        "Locally suspicious",
        local_only_flagged,
    )

    metric_columns[4].metric(
        "Final flagged",
        final_flagged,
    )

    if not openrouter_key and suspicious:
        st.warning(
            f"{len(suspicious)} chunk(s) crossed the local threshold. "
            "Add OPENROUTER_API_KEY to semantically review only those "
            "transcript chunks."
        )

    elif (
        openrouter_key
        and len(suspicious) > review_count
    ):
        st.warning(
            f"Only {review_count} of {len(suspicious)} suspicious chunks "
            "were reviewed because of the configured review limit."
        )

    with st.expander("Keyword hits", expanded=True):
        render_keyword_hits(chunks)

    with st.expander("Full local transcript"):
        st.write(
            " ".join(
                chunk.text
                for chunk in chunks
            )
            or "No speech detected."
        )

    st.subheader("Chunk details")
    render_results(chunks)

    report = {
        "app": APP_TITLE,
        "detected_language": detected_language,
        "duration_seconds": duration,
        "local_model": model_size,
        "chunk_seconds": chunk_seconds,
        "local_threshold": local_threshold,
        "keyword_min_probability": keyword_min_probability,
        "fuzzy_keywords_enabled": enable_fuzzy_keywords,
        "fuzzy_ratio": fuzzy_ratio,
        "openrouter_model": (
            openrouter_model
            if openrouter_key
            else None
        ),
        "openrouter_chunks_reviewed": (
            review_count
            if openrouter_key
            else 0
        ),
        "chunks": [
            chunk.as_dict()
            for chunk in chunks
        ],
    }

    st.download_button(
        "Download JSON report",
        data=json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        file_name="voice_nsfw_report.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()