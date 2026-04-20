"""
Tool loader — imports all tool modules so they self-register with the registry.

Call `load_all_tools()` once at startup (done by AgentEngine automatically).
Adding a new tool module? Just add its import here.
"""

from __future__ import annotations

from utils.logger import get_logger

log = get_logger(__name__)


def load_all_tools() -> int:
    """
    Import every tool module. Each module registers its tools on import.
    Returns the total number of registered tools.
    """
    import tools.system_tools   # noqa: F401
    import tools.file_tools     # noqa: F401
    import tools.git_tools      # noqa: F401
    import tools.email_tool     # noqa: F401
    import tools.desktop_tools  # noqa: F401

    from tools.registry import registry
    count = len(registry)
    log.info("Tools loaded", extra={"count": count, "tools": [t.name for t in registry.list_tools()]})
    return count
