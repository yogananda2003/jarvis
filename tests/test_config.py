"""
Tests for config loading and the tool registry.
Run with:  python -m pytest tests/ -v
"""

import os
import sys
from pathlib import Path

# Ensure project root is on the path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestSettings:
    def test_settings_loads(self):
        from config import settings
        assert settings is not None

    def test_default_llm_model(self):
        from config import settings
        assert settings.llm.model == "llama3.2:1b"

    def test_default_api_port(self):
        from config import settings
        assert settings.api.port == 8000

    def test_project_root_exists(self):
        from config import settings
        assert settings.project_root.is_dir()

    def test_env_override(self, monkeypatch):
        """JARVIS_LLM__MODEL env var should override the YAML value."""
        monkeypatch.setenv("JARVIS_LLM__MODEL", "mistral")
        # Re-import to pick up the override (reload the module)
        import importlib
        import config.config as cfg_mod
        cfg_mod._configured = False  # allow re-run of _load_settings in isolation
        raw = {"llm": {"model": "llama3"}}
        raw = cfg_mod._apply_env_overrides(raw)
        assert raw["llm"]["model"] == "mistral"


# ---------------------------------------------------------------------------
# Tool registry tests
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_singleton(self):
        from tools.registry import ToolRegistry
        r1 = ToolRegistry()
        r2 = ToolRegistry()
        assert r1 is r2

    def test_register_and_execute(self):
        from tools.registry import ToolRegistry, ToolParameter
        registry = ToolRegistry()

        @registry.register(
            name="_test_add",
            description="Add two numbers",
            category="test",
            parameters=[
                ToolParameter("a", "integer", "first operand"),
                ToolParameter("b", "integer", "second operand"),
            ],
        )
        def _add(a: int, b: int) -> int:
            return a + b

        result = registry.execute("_test_add", a=3, b=4)
        assert result == 7

    def test_disabled_tool_raises(self):
        from tools.registry import ToolRegistry
        import pytest
        registry = ToolRegistry()
        registry.disable("_test_add")
        with pytest.raises(RuntimeError, match="disabled"):
            registry.execute("_test_add", a=1, b=2)

    def test_schema_output(self):
        from tools.registry import ToolRegistry
        registry = ToolRegistry()
        registry.enable("_test_add")
        schemas = registry.all_schemas()
        names = [s["name"] for s in schemas]
        assert "_test_add" in names

    def test_unknown_tool_raises(self):
        from tools.registry import ToolRegistry
        import pytest
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.execute("__nonexistent__")
