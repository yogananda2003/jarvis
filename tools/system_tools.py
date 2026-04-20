"""
System tools — open/close apps, system info, process management.

All tools self-register with the global ToolRegistry on import.
Dangerous operations (shell exec, process kill) are gated by config flags.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional

import psutil

from config import settings
from tools.registry import ToolParameter, registry
from utils.logger import get_logger

log = get_logger(__name__)

_OS = platform.system()   # "Windows" | "Darwin" | "Linux"


# ---------------------------------------------------------------------------
# open_app
# ---------------------------------------------------------------------------

# Common app name → shell command mapping for Windows
_WIN_ALIASES: Dict[str, str] = {
    # Browsers
    "chrome":               "chrome",
    "google chrome":        "chrome",
    "firefox":              "firefox",
    "edge":                 "msedge",
    "microsoft edge":       "msedge",
    "brave":                "brave",
    # Editors / IDEs
    "notepad":              "notepad",
    "notepad++":            "notepad++",
    "vscode":               "code",
    "vs code":              "code",
    "visual studio code":   "code",
    "sublime":              "sublime_text",
    # Office
    "word":                 "winword",
    "excel":                "excel",
    "powerpoint":           "powerpnt",
    "outlook":              "outlook",
    # System
    "explorer":             "explorer",
    "file explorer":        "explorer",
    "calculator":           "calc",
    "terminal":             "wt",
    "windows terminal":     "wt",
    "cmd":                  "cmd",
    "powershell":           "powershell",
    "task manager":         "taskmgr",
    "control panel":        "control",
    "paint":                "mspaint",
    "snipping tool":        "SnippingTool",
    # Communication
    "teams":                "teams",
    "slack":                "slack",
    "discord":              "discord",
    "zoom":                 "zoom",
    "whatsapp":             "whatsapp",
    # Entertainment
    "spotify":              "spotify",
    "vlc":                  "vlc",
    "steam":                "steam",
    "obs":                  "obs64",
    "obs studio":           "obs64",
}

# Window title keywords for foreground activation
_WIN_TITLE_HINTS: Dict[str, str] = {
    "chrome":       "Chrome",
    "firefox":      "Firefox",
    "msedge":       "Edge",
    "edge":         "Edge",
    "notepad":      "Notepad",
    "notepad++":    "Notepad++",
    "code":         "Visual Studio Code",
    "calc":         "Calculator",
    "calculator":   "Calculator",
    "mspaint":      "Paint",
    "taskmgr":      "Task Manager",
    "discord":      "Discord",
    "spotify":      "Spotify",
    "steam":        "Steam",
    "slack":        "Slack",
    "teams":        "Teams",
    "obs64":        "OBS",
    "explorer":     "File Explorer",
    "vlc":          "VLC",
    "zoom":         "Zoom",
    "winword":      "Word",
    "excel":        "Excel",
    "powerpnt":     "PowerPoint",
}


def _activate_window_foreground(app_name: str, resolved: str) -> None:
    """Wait for a newly launched app to appear, then bring it to the foreground."""
    import time
    time.sleep(1.2)  # give the app time to start

    hint = (
        _WIN_TITLE_HINTS.get(resolved.lower())
        or _WIN_TITLE_HINTS.get(app_name.lower())
        or app_name
    )

    try:
        import pygetwindow as gw
        for _ in range(3):  # retry up to 3 times
            matches = [w for w in gw.getAllWindows()
                       if hint.lower() in w.title.lower() and w.title.strip()]
            if matches:
                win = matches[0]
                if win.isMinimized:
                    win.restore()
                win.maximize()
                win.activate()
                log.info("Window activated", extra={"title": win.title})
                return
            time.sleep(0.8)
    except Exception as exc:
        log.debug("Window activation failed", extra={"error": str(exc)})


# macOS/Linux aliases
_UNIX_ALIASES: Dict[str, str] = {
    "chrome":               "google-chrome",
    "google chrome":        "google-chrome",
    "firefox":              "firefox",
    "vscode":               "code",
    "vs code":              "code",
    "terminal":             "gnome-terminal",
    "calculator":           "gnome-calculator",
    "files":                "nautilus",
    "spotify":              "spotify",
    "discord":              "discord",
}


@registry.register(
    name="open_app",
    description="Launch a desktop application by name (e.g. 'chrome', 'notepad', 'spotify', 'discord', 'vscode').",
    category="system",
    parameters=[
        ToolParameter("app_name", "string", "Name of the application to open", required=True),
    ],
    enabled=settings.tools.system.enabled,
)
def open_app(app_name: str) -> str:
    log.info("Opening application", extra={"app": app_name})
    key = app_name.strip().lower()

    if _OS == "Windows":
        resolved = _WIN_ALIASES.get(key, app_name)
        # ms-settings: and other URI schemes — use os.startfile directly
        if ":" in resolved and not resolved.endswith(".exe"):
            try:
                os.startfile(resolved)
                return f"Opened '{app_name}' successfully."
            except Exception as exc:
                return f"Failed to open '{app_name}': {exc}"
        # Use Windows shell 'start' which searches PATH + App Paths registry
        try:
            subprocess.Popen(
                f'start "" "{resolved}"',
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Bring the window to foreground after a brief launch delay
            _activate_window_foreground(app_name, resolved)
            return f"Launched '{app_name}' successfully."
        except Exception as exc:
            return f"Failed to launch '{app_name}': {exc}"
    else:
        resolved = _UNIX_ALIASES.get(key, app_name)
        try:
            subprocess.Popen(
                [resolved],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"Launched '{app_name}' successfully."
        except FileNotFoundError:
            return f"Application '{app_name}' not found. Make sure it is installed."
        except Exception as exc:
            return f"Failed to open '{app_name}': {exc}"


# ---------------------------------------------------------------------------
# get_system_info
# ---------------------------------------------------------------------------

@registry.register(
    name="get_system_info",
    description="Return CPU usage, RAM usage, disk space, and OS information.",
    category="system",
    parameters=[],
    enabled=settings.tools.system.enabled,
)
def get_system_info() -> Dict[str, Any]:
    cpu_pct = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu_percent": cpu_pct,
        "ram_total_gb": round(ram.total / 1e9, 1),
        "ram_used_gb": round(ram.used / 1e9, 1),
        "ram_percent": ram.percent,
        "disk_total_gb": round(disk.total / 1e9, 1),
        "disk_used_gb": round(disk.used / 1e9, 1),
        "disk_percent": disk.percent,
        "python_version": sys.version.split()[0],
    }

    log.info("System info retrieved", extra=info)
    return info


# ---------------------------------------------------------------------------
# list_processes
# ---------------------------------------------------------------------------

@registry.register(
    name="list_processes",
    description="List running processes. Optionally filter by name.",
    category="system",
    parameters=[
        ToolParameter("filter_name", "string", "Optional substring to filter process names", required=False, default=""),
    ],
    enabled=settings.tools.system.enabled,
)
def list_processes(filter_name: str = "") -> List[Dict[str, Any]]:
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            info = proc.info
            if filter_name and filter_name.lower() not in info["name"].lower():
                continue
            processes.append({
                "pid": info["pid"],
                "name": info["name"],
                "cpu_pct": round(info["cpu_percent"] or 0.0, 1),
                "mem_pct": round(info["memory_percent"] or 0.0, 2),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    processes.sort(key=lambda p: p["cpu_pct"], reverse=True)
    return processes[:50]   # cap at 50 to avoid overwhelming the LLM


# ---------------------------------------------------------------------------
# kill_process  (permission-gated)
# ---------------------------------------------------------------------------

@registry.register(
    name="kill_process",
    description="Kill a running process by name or PID. Requires explicit permission.",
    category="system",
    parameters=[
        ToolParameter("identifier", "string", "Process name (e.g. 'chrome.exe') or PID as string"),
    ],
    enabled=settings.tools.system.enabled,
)
def kill_process(identifier: str) -> str:
    if not settings.tools.system.allow_shell_exec:
        return (
            "kill_process is disabled. "
            "Set tools.system.allow_shell_exec=true in config.yaml to enable."
        )

    log.warning("Killing process", extra={"identifier": identifier})

    # Try as PID first, then as name
    try:
        pid = int(identifier)
        proc = psutil.Process(pid)
        proc.terminate()
        return f"Sent SIGTERM to process PID {pid} ({proc.name()})."
    except ValueError:
        pass   # not a number — try name match

    killed = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if identifier.lower() in proc.info["name"].lower():
                proc.terminate()
                killed.append(f"{proc.info['name']} (PID {proc.info['pid']})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if killed:
        return f"Terminated: {', '.join(killed)}"
    return f"No process matching '{identifier}' found."
