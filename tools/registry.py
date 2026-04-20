"""
Tool Registry — the single source of truth for all JARVIS tools.

Tools register themselves here at import time via the @registry.register
decorator. The agent looks up tools by name when building its action plan.

Design principles:
- Zero coupling to agent internals
- Tools declare their own schema (name, description, parameters)
- The registry is a simple dict under the hood — no magic
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ToolParameter:
    """Describes a single parameter that a tool accepts."""
    name: str
    type: str               # "string" | "integer" | "boolean" | "object"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Full definition of a registered tool."""
    name: str
    description: str
    category: str           # "system" | "email" | "git" | "calendar" | ...
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable[..., Any]] = field(default=None, repr=False)
    enabled: bool = True

    def to_schema(self) -> dict:
        """Return an OpenAI-style function schema for LLM tool-calling."""
        props: dict = {}
        required: list = []
        for p in self.parameters:
            props[p.name] = {"type": p.type, "description": p.description}
            if p.default is not None:
                props[p.name]["default"] = p.default
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }

    def execute(self, **kwargs: Any) -> Any:
        """Invoke the tool handler with validated keyword arguments."""
        if not self.enabled:
            raise RuntimeError(f"Tool '{self.name}' is disabled.")
        if self.handler is None:
            raise NotImplementedError(f"Tool '{self.name}' has no handler registered.")
        return self.handler(**kwargs)


class ToolRegistry:
    """Central registry mapping tool names to their definitions."""

    _instance: Optional["ToolRegistry"] = None

    def __new__(cls) -> "ToolRegistry":
        # Singleton — one registry for the entire process lifetime
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, ToolDefinition] = {}
        return cls._instance

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        category: str,
        parameters: Optional[List[ToolParameter]] = None,
        enabled: bool = True,
    ) -> Callable:
        """
        Decorator factory for registering a function as a JARVIS tool.

        Usage:
            @registry.register(
                name="open_app",
                description="Launch a desktop application by name.",
                category="system",
                parameters=[ToolParameter("app_name", "string", "Name of the app")],
            )
            def open_app(app_name: str) -> str:
                ...
        """
        def decorator(fn: Callable) -> Callable:
            tool = ToolDefinition(
                name=name,
                description=description,
                category=category,
                parameters=parameters or [],
                handler=fn,
                enabled=enabled,
            )
            if name in self._tools:
                log.warning("Tool already registered — overwriting", extra={"tool": name})
            self._tools[name] = tool
            log.debug("Tool registered", extra={"tool": name, "category": category})
            return fn

        return decorator

    def enable(self, name: str) -> None:
        self._get(name).enabled = True

    def disable(self, name: str) -> None:
        self._get(name).enabled = False

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def _get(self, name: str) -> ToolDefinition:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: '{name}'")
        return tool

    def list_tools(self, category: Optional[str] = None) -> List[ToolDefinition]:
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def list_enabled(self) -> List[ToolDefinition]:
        return [t for t in self._tools.values() if t.enabled]

    def all_schemas(self) -> List[dict]:
        """Return OpenAI-style schemas for all enabled tools (for LLM context)."""
        return [t.to_schema() for t in self.list_enabled()]

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Look up and execute a tool by name, recording metrics."""
        from utils.metrics import metrics
        with metrics.timer("tool_execution_ms"):
            try:
                result = self._get(name).execute(**kwargs)
                metrics.inc("tool_calls_total")
                return result
            except Exception:
                metrics.inc("tool_errors_total")
                raise

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        names = list(self._tools.keys())
        return f"<ToolRegistry tools={names}>"


# Module-level singleton — import this everywhere
registry = ToolRegistry()
