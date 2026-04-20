"""
Desktop automation tools — type text, press hotkeys, open URLs.

Requires: pyautogui
"""

from __future__ import annotations

import subprocess
import webbrowser
from typing import Optional

from config import settings
from tools.registry import ToolParameter, registry
from utils.logger import get_logger

log = get_logger(__name__)

_ENABLED = settings.tools.system.enabled


@registry.register(
    name="type_text",
    description="Type text at the current cursor position (simulates keyboard input).",
    category="desktop",
    parameters=[
        ToolParameter("text", "string", "Text to type", required=True),
        ToolParameter("interval", "number", "Delay between keystrokes in seconds (default 0.05)", required=False, default=0.05),
    ],
    enabled=_ENABLED,
)
def type_text(text: str, interval: float = 0.05) -> str:
    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.write(text, interval=interval)
        log.info("Typed text", extra={"length": len(text)})
        return f"Typed {len(text)} characters successfully."
    except ImportError:
        return "pyautogui is not installed. Run: pip install pyautogui"
    except Exception as exc:
        return f"Failed to type text: {exc}"


@registry.register(
    name="press_hotkey",
    description="Press a keyboard shortcut or hotkey (e.g. 'ctrl+c', 'win+d', 'alt+f4', 'enter', 'escape').",
    category="desktop",
    parameters=[
        ToolParameter("keys", "string", "Key combination separated by '+' (e.g. 'ctrl+c', 'win+r')", required=True),
    ],
    enabled=_ENABLED,
)
def press_hotkey(keys: str) -> str:
    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        parts = [k.strip() for k in keys.lower().split("+")]
        if len(parts) == 1:
            pyautogui.press(parts[0])
        else:
            pyautogui.hotkey(*parts)
        log.info("Pressed hotkey", extra={"keys": keys})
        return f"Pressed '{keys}' successfully."
    except ImportError:
        return "pyautogui is not installed. Run: pip install pyautogui"
    except Exception as exc:
        return f"Failed to press hotkey '{keys}': {exc}"


@registry.register(
    name="open_website",
    description="Open a website URL in the default browser.",
    category="desktop",
    parameters=[
        ToolParameter("url", "string", "URL to open (include https://)", required=True),
    ],
    enabled=_ENABLED,
)
def open_website(url: str) -> str:
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        webbrowser.open(url)
        log.info("Opened website", extra={"url": url})
        return f"Opened '{url}' in default browser."
    except Exception as exc:
        return f"Failed to open website: {exc}"


@registry.register(
    name="focus_window",
    description="Bring a window to focus by its title (partial match). Windows only.",
    category="desktop",
    parameters=[
        ToolParameter("title", "string", "Window title or partial title to focus", required=True),
    ],
    enabled=_ENABLED,
)
def focus_window(title: str) -> str:
    try:
        import pygetwindow as gw
        matches = gw.getWindowsWithTitle(title)
        if not matches:
            return f"No window found matching '{title}'."
        win = matches[0]
        win.activate()
        log.info("Focused window", extra={"title": win.title})
        return f"Focused window: '{win.title}'"
    except ImportError:
        return "pygetwindow is not installed (comes with pyautogui). Run: pip install pyautogui"
    except Exception as exc:
        return f"Failed to focus window '{title}': {exc}"
