"""
Configuration loader for JARVIS.

Loads config.yaml, overlays environment variables (prefixed JARVIS_),
and exposes a single typed `settings` singleton used across the codebase.

Environment variable override convention:
    JARVIS_LLM__MODEL=mistral  →  settings.llm.model = "mistral"
    JARVIS_API__PORT=9000      →  settings.api.port = 9000
    (double-underscore = nesting separator)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Locate project root regardless of where the process is launched from
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_FILE = _PROJECT_ROOT / "config" / "config.yaml"
_ENV_FILE = _PROJECT_ROOT / ".env"

load_dotenv(_ENV_FILE)


# ---------------------------------------------------------------------------
# Typed sub-configs (dataclasses for IDE support + runtime validation)
# ---------------------------------------------------------------------------

@dataclass
class RetryConfig:
    max_attempts: int = 3
    backoff_factor: int = 2


@dataclass
class LLMConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 2048
    streaming: bool = True
    timeout: int = 60
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class STTConfig:
    engine: str = "whisper"
    model_size: str = "base"
    language: str = "en"
    device: str = "cpu"


@dataclass
class TTSConfig:
    engine: str = "pyttsx3"
    rate: int = 175
    volume: float = 1.0
    voice_id: Optional[str] = None


@dataclass
class WakeWordConfig:
    enabled: bool = True
    keyword: str = "jarvis"
    sensitivity: float = 0.5
    engine: str = "whisper"    # "whisper" | "porcupine"


@dataclass
class VoiceConfig:
    enabled: bool = True
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)


@dataclass
class AgentConfig:
    planner_model: str = "llama3"
    executor_model: str = "llama3"
    max_iterations: int = 10
    timeout: int = 120
    verbose: bool = False


@dataclass
class ShortTermMemoryConfig:
    max_messages: int = 20


@dataclass
class LongTermMemoryConfig:
    backend: str = "sqlite"
    db_path: str = "data/memory.db"
    embeddings_enabled: bool = False


@dataclass
class MemoryConfig:
    short_term: ShortTermMemoryConfig = field(default_factory=ShortTermMemoryConfig)
    long_term: LongTermMemoryConfig = field(default_factory=LongTermMemoryConfig)


@dataclass
class SystemToolsConfig:
    enabled: bool = True
    allow_file_delete: bool = False
    allow_shell_exec: bool = False


@dataclass
class EmailToolConfig:
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    use_tls: bool = True


@dataclass
class ToolsConfig:
    system: SystemToolsConfig = field(default_factory=SystemToolsConfig)
    email: EmailToolConfig = field(default_factory=EmailToolConfig)
    git_enabled: bool = True
    calendar_enabled: bool = False


@dataclass
class APIConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"
    log_dir: str = "logs"
    max_bytes: int = 10_485_760
    backup_count: int = 5
    audit_log: bool = True


@dataclass
class FeaturesConfig:
    voice: bool = True
    api_server: bool = True
    long_term_memory: bool = True
    multi_agent: bool = False
    web_automation: bool = False
    plugin_system: bool = False


@dataclass
class JarvisMetaConfig:
    name: str = "JARVIS"
    version: str = "0.1.0"
    debug: bool = False


@dataclass
class Settings:
    jarvis: JarvisMetaConfig = field(default_factory=JarvisMetaConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)

    # Resolved absolute paths (not in YAML)
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)

    @property
    def debug(self) -> bool:
        return self.jarvis.debug

    @property
    def long_term_db_path(self) -> Path:
        return self.project_root / self.memory.long_term.db_path

    @property
    def log_dir_path(self) -> Path:
        return self.project_root / self.logging.log_dir

    def validate(self) -> list[str]:
        """
        Validate the loaded settings. Returns a list of error strings.
        An empty list means the config is valid.
        """
        errors: list[str] = []

        if not (1 <= self.api.port <= 65535):
            errors.append(f"api.port={self.api.port} must be between 1 and 65535")

        if not self.llm.base_url.startswith(("http://", "https://")):
            errors.append(f"llm.base_url must start with http:// or https://")

        if self.llm.max_tokens < 1:
            errors.append(f"llm.max_tokens must be >= 1")

        if self.llm.temperature < 0.0 or self.llm.temperature > 2.0:
            errors.append(f"llm.temperature must be between 0.0 and 2.0")

        if self.agent.max_iterations < 1:
            errors.append(f"agent.max_iterations must be >= 1")

        if self.memory.short_term.max_messages < 1:
            errors.append(f"memory.short_term.max_messages must be >= 1")

        if self.logging.level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append(f"logging.level='{self.logging.level}' is not a valid log level")

        return errors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(raw: dict) -> dict:
    """
    Map JARVIS_SECTION__KEY=value env vars onto the raw config dict.
    Double-underscore separates nesting levels.
    """
    prefix = "JARVIS_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("__")
        target = raw
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        leaf_key = parts[-1]
        # Attempt type coercion: bool > int > float > str
        if env_val.lower() in ("true", "false"):
            target[leaf_key] = env_val.lower() == "true"
        else:
            for cast in (int, float):
                try:
                    target[leaf_key] = cast(env_val)
                    break
                except ValueError:
                    pass
            else:
                target[leaf_key] = env_val
    return raw


def _dict_to_settings(raw: dict) -> Settings:
    """Hydrate a flat dict (from YAML) into the Settings dataclass tree."""

    def _get(d: dict, *keys, default=None):
        node = d
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
        return node

    j = raw.get("jarvis", {})
    l = raw.get("llm", {})
    v = raw.get("voice", {})
    a = raw.get("agent", {})
    m = raw.get("memory", {})
    t = raw.get("tools", {})
    api = raw.get("api", {})
    log = raw.get("logging", {})
    feat = raw.get("features", {})

    return Settings(
        jarvis=JarvisMetaConfig(
            name=j.get("name", "JARVIS"),
            version=j.get("version", "0.1.0"),
            debug=j.get("debug", False),
        ),
        llm=LLMConfig(
            provider=l.get("provider", "ollama"),
            base_url=l.get("base_url", "http://localhost:11434"),
            model=l.get("model", "llama3"),
            temperature=l.get("temperature", 0.7),
            max_tokens=l.get("max_tokens", 2048),
            streaming=l.get("streaming", True),
            timeout=l.get("timeout", 60),
            retry=RetryConfig(**l.get("retry", {})),
        ),
        voice=VoiceConfig(
            enabled=v.get("enabled", True),
            stt=STTConfig(**v.get("stt", {})),
            tts=TTSConfig(**v.get("tts", {})),
            wake_word=WakeWordConfig(**v.get("wake_word", {})),
        ),
        agent=AgentConfig(**a),
        memory=MemoryConfig(
            short_term=ShortTermMemoryConfig(**m.get("short_term", {})),
            long_term=LongTermMemoryConfig(**m.get("long_term", {})),
        ),
        tools=ToolsConfig(
            system=SystemToolsConfig(**t.get("system", {})),
            email=EmailToolConfig(**t.get("email", {})),
            git_enabled=t.get("git", {}).get("enabled", True),
            calendar_enabled=t.get("calendar", {}).get("enabled", False),
        ),
        api=APIConfig(
            enabled=api.get("enabled", True),
            host=api.get("host", "127.0.0.1"),
            port=api.get("port", 8000),
            cors_origins=api.get("cors_origins", ["http://localhost:3000"]),
        ),
        logging=LoggingConfig(
            level=log.get("level", "INFO"),
            format=log.get("format", "json"),
            log_dir=log.get("log_dir", "logs"),
            max_bytes=log.get("max_bytes", 10_485_760),
            backup_count=log.get("backup_count", 5),
            audit_log=log.get("audit_log", True),
        ),
        features=FeaturesConfig(
            voice=feat.get("voice", True),
            api_server=feat.get("api_server", True),
            long_term_memory=feat.get("long_term_memory", True),
            multi_agent=feat.get("multi_agent", False),
            web_automation=feat.get("web_automation", False),
            plugin_system=feat.get("plugin_system", False),
        ),
    )


# ---------------------------------------------------------------------------
# Public singleton
# ---------------------------------------------------------------------------

def _load_settings() -> Settings:
    if not _CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {_CONFIG_FILE}")

    with open(_CONFIG_FILE, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    raw = _apply_env_overrides(raw)
    cfg = _dict_to_settings(raw)

    # Ensure runtime directories exist
    cfg.log_dir_path.mkdir(parents=True, exist_ok=True)
    cfg.long_term_db_path.parent.mkdir(parents=True, exist_ok=True)

    return cfg


settings: Settings = _load_settings()
