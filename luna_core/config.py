"""Typed configuration for Luna, loaded once from config.yaml.

Replaces the jarvis-era pattern of re-reading and re-parsing the YAML file
from inside every module. Call `load_config()` once at startup; everything
else calls `get_config()`. `reload_config()` / `save_config()` back the
settings REST API in server.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config.yaml"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8756


class WakeWordConfig(BaseModel):
    enabled: bool = False
    model_path: str | None = None
    phrase: str = "hey luna"
    threshold: float = 0.5
    chunk_size: int = 1280
    cooldown_seconds: float = 2.0
    interrupt_cooldown_seconds: float = 3.0


class SttConfig(BaseModel):
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    sample_rate: int = 16000
    silence_threshold: float = 0.012
    speech_threshold: float = 0.015
    silent_chunks_needed: int = 7
    max_seconds: int = 30


class TtsConfig(BaseModel):
    engine: Literal["piper"] = "piper"
    voice: str = "en_US-lessac-high"
    voice_model_path: str = "luna_core/data/voices/en_US-lessac-high.onnx"
    speed: float = 1.0
    sample_rate: int = 22050
    piper_executable: str | None = None


class ProviderConfig(BaseModel):
    type: Literal["ollama", "openai"] = "ollama"
    label: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""


class LlmConfig(BaseModel):
    active_provider: str = "ollama"
    temperature: float = 0.7
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)


class ContextConfig(BaseModel):
    max_messages: int = 10


class PermissionsConfig(BaseModel):
    default_level: Literal["allow", "deny", "ask"] = "deny"
    tools: dict[str, Literal["allow", "deny", "ask"]] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    dir: str = "luna_core/data/logs"


class LunaConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


_config: LunaConfig | None = None


def load_config(path: Path | None = None) -> LunaConfig:
    """Load and cache config.yaml. Safe to call more than once (e.g. tests)."""
    global _config
    cfg_path = path or _DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    _config = LunaConfig.model_validate(raw)
    return _config


def get_config() -> LunaConfig:
    """Return the cached config, loading it from disk on first access."""
    if _config is None:
        return load_config()
    return _config


def reload_config(path: Path | None = None) -> LunaConfig:
    """Force a fresh read from disk, discarding the cached config."""
    return load_config(path)


def save_config(cfg: LunaConfig, path: Path | None = None) -> None:
    global _config
    cfg_path = path or _DEFAULT_CONFIG_PATH
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg.model_dump(), f, default_flow_style=False, sort_keys=False)
    _config = cfg
