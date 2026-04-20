"""
File system tools — read, write, list, search files.

Delete is permission-gated via config.yaml:
    tools.system.allow_file_delete: true
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings
from tools.registry import ToolParameter, registry
from utils.logger import get_logger

log = get_logger(__name__)

_MAX_READ_BYTES = 50_000     # cap file reads to avoid flooding the LLM context


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

@registry.register(
    name="read_file",
    description="Read and return the contents of a text file.",
    category="file",
    parameters=[
        ToolParameter("path", "string", "Absolute or relative path to the file"),
        ToolParameter("max_lines", "integer", "Maximum number of lines to return (default 200)", required=False, default=200),
    ],
    enabled=settings.tools.system.enabled,
)
def read_file(path: str, max_lines: int = 200) -> str:
    p = Path(path).expanduser().resolve()
    log.info("Reading file", extra={"path": str(p)})

    if not p.exists():
        return f"File not found: {p}"
    if not p.is_file():
        return f"Path is not a file: {p}"
    if p.stat().st_size > _MAX_READ_BYTES * 2:
        return (
            f"File is too large to read in full ({p.stat().st_size:,} bytes). "
            "Use a more specific path or a search tool."
        )

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) > max_lines:
            truncated = len(lines) - max_lines
            lines = lines[:max_lines]
            lines.append(f"\n... [{truncated} more lines truncated] ...")
        return "\n".join(lines)
    except PermissionError:
        return f"Permission denied: {p}"
    except Exception as exc:
        return f"Error reading file: {exc}"


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

@registry.register(
    name="write_file",
    description="Write text content to a file. Creates parent directories if needed.",
    category="file",
    parameters=[
        ToolParameter("path", "string", "Absolute or relative path to write to"),
        ToolParameter("content", "string", "Text content to write"),
        ToolParameter("append", "boolean", "If true, append to existing file instead of overwriting", required=False, default=False),
    ],
    enabled=settings.tools.system.enabled,
)
def write_file(path: str, content: str, append: bool = False) -> str:
    p = Path(path).expanduser().resolve()
    log.info("Writing file", extra={"path": str(p), "append": append})

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(p, mode, encoding="utf-8") as f:
            f.write(content)
        action = "Appended to" if append else "Wrote"
        return f"{action} '{p}' ({len(content):,} characters)."
    except PermissionError:
        return f"Permission denied: {p}"
    except Exception as exc:
        return f"Error writing file: {exc}"


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

@registry.register(
    name="list_directory",
    description="List the contents of a directory.",
    category="file",
    parameters=[
        ToolParameter("path", "string", "Directory path to list (defaults to current directory)", required=False, default="."),
        ToolParameter("show_hidden", "boolean", "Include hidden files/folders", required=False, default=False),
    ],
    enabled=settings.tools.system.enabled,
)
def list_directory(path: str = ".", show_hidden: bool = False) -> List[Dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    log.info("Listing directory", extra={"path": str(p)})

    if not p.exists():
        return [{"error": f"Path not found: {p}"}]
    if not p.is_dir():
        return [{"error": f"Not a directory: {p}"}]

    items = []
    try:
        for entry in sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
            if not show_hidden and entry.name.startswith("."):
                continue
            stat = entry.stat()
            items.append({
                "name": entry.name,
                "type": "file" if entry.is_file() else "dir",
                "size_bytes": stat.st_size if entry.is_file() else None,
            })
    except PermissionError:
        return [{"error": f"Permission denied: {p}"}]

    return items


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------

@registry.register(
    name="search_files",
    description="Recursively search for files matching a name pattern.",
    category="file",
    parameters=[
        ToolParameter("pattern", "string", "Glob pattern, e.g. '*.py' or 'config.*'"),
        ToolParameter("root", "string", "Root directory to search from (defaults to current directory)", required=False, default="."),
        ToolParameter("max_results", "integer", "Maximum number of results to return", required=False, default=20),
    ],
    enabled=settings.tools.system.enabled,
)
def search_files(pattern: str, root: str = ".", max_results: int = 20) -> List[str]:
    root_path = Path(root).expanduser().resolve()
    log.info("Searching files", extra={"pattern": pattern, "root": str(root_path)})

    matches = []
    try:
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Skip hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if fnmatch.fnmatch(filename, pattern):
                    full = os.path.join(dirpath, filename)
                    matches.append(full)
                    if len(matches) >= max_results:
                        return matches
    except PermissionError as exc:
        log.warning("Search permission error", extra={"error": str(exc)})

    return matches


# ---------------------------------------------------------------------------
# delete_file  (permission-gated)
# ---------------------------------------------------------------------------

@registry.register(
    name="delete_file",
    description="Delete a file. Requires explicit permission in config.",
    category="file",
    parameters=[
        ToolParameter("path", "string", "Absolute or relative path to the file to delete"),
    ],
    enabled=settings.tools.system.enabled,
)
def delete_file(path: str) -> str:
    if not settings.tools.system.allow_file_delete:
        return (
            "delete_file is disabled for safety. "
            "Set tools.system.allow_file_delete=true in config.yaml to enable."
        )

    p = Path(path).expanduser().resolve()
    log.warning("Deleting file", extra={"path": str(p)})

    if not p.exists():
        return f"File not found: {p}"
    if p.is_dir():
        return f"Path is a directory, not a file: {p}. Use a directory removal tool."

    try:
        p.unlink()
        return f"Deleted: {p}"
    except PermissionError:
        return f"Permission denied: {p}"
    except Exception as exc:
        return f"Error deleting file: {exc}"
