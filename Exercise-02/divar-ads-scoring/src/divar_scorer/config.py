from __future__ import annotations

import os
from dataclasses import dataclass, field


def _csv_env(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass(slots=True)
class Settings:
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "").strip())
    openrouter_url: str = "https://api.gapgpt.app/v1/chat/completions"
    light_models: list[str] = field(
        default_factory=lambda: _csv_env(
            "OPENROUTER_LIGHT_MODELS",
            "openai/gpt-5-mini,qwen/qwen3.6-35b-a3b",
        )
    )
    heavy_models: list[str] = field(
        default_factory=lambda: _csv_env(
            "OPENROUTER_HEAVY_MODELS",
            "anthropic/claude-sonnet-4.5,openai/gpt-5",
        )
    )
    app_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_APP_URL", "http://localhost:8501"))
    app_name: str = field(default_factory=lambda: os.getenv("OPENROUTER_APP_NAME", "DivarLens"))
    clip_model_id: str = field(
        default_factory=lambda: os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
    )
    request_timeout_seconds: float = 75.0
