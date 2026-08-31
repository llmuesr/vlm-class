from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .models import ImageSignals


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _resolution_score(image: Image.Image) -> float:
    width, height = image.size
    short_side = min(width, height)
    megapixels = (width * height) / 1_000_000
    side_score = _clamp((short_side - 240) / (1080 - 240) * 100)
    mp_score = _clamp((megapixels - 0.15) / (2.0 - 0.15) * 100)
    return 0.65 * side_score + 0.35 * mp_score


def _exposure_score(gray: np.ndarray) -> float:
    mean = float(gray.mean())
    clipped = float(((gray < 8) | (gray > 247)).mean())
    center_score = _clamp(100 - abs(mean - 127.5) / 1.15)
    return _clamp(center_score - clipped * 140)


def _sharpness_score(gray: np.ndarray) -> float:
    # Mean absolute spatial gradient is inexpensive and dependency-free.
    dx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    dy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    gradient = float((dx + dy) / 2)
    # Natural marketplace photos are usually usable above ~12 and crisp above ~28.
    return _clamp((gradient - 3.5) / 24.5 * 100)


def analyze_image_quality(images: list[Image.Image]) -> ImageSignals:
    if not images:
        return ImageSignals(warnings=["No images were supplied."])

    resolution_scores: list[float] = []
    exposure_scores: list[float] = []
    sharpness_scores: list[float] = []
    per_image: list[float] = []
    warnings: list[str] = []

    for index, source in enumerate(images):
        image = source.convert("RGB")
        thumb = image.copy()
        thumb.thumbnail((640, 640))
        gray = np.asarray(thumb.convert("L"), dtype=np.float32)
        resolution = _resolution_score(image)
        exposure = _exposure_score(gray)
        sharpness = _sharpness_score(gray)
        score = 0.42 * resolution + 0.28 * exposure + 0.30 * sharpness

        resolution_scores.append(round(resolution, 1))
        exposure_scores.append(round(exposure, 1))
        sharpness_scores.append(round(sharpness, 1))
        per_image.append(round(score, 1))

        label = f"Image {index + 1}"
        if resolution < 40:
            warnings.append(f"{label} has low resolution ({image.width}×{image.height}).")
        if exposure < 40:
            warnings.append(f"{label} is very dark, bright, or heavily clipped.")
        if sharpness < 32:
            warnings.append(f"{label} may be blurry or lack detail.")
        if not math.isfinite(score):
            per_image[-1] = 0.0

    return ImageSignals(
        quality_score=round(float(np.mean(per_image)), 1),
        per_image_quality=per_image,
        resolution_scores=resolution_scores,
        exposure_scores=exposure_scores,
        sharpness_scores=sharpness_scores,
        warnings=warnings,
    )

