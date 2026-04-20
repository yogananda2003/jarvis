"""
Git tools — status, log, diff, commit, branch management.

All tools run `git` as a subprocess and parse the output.
The working directory defaults to the current directory but can be overridden.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings
from tools.registry import ToolParameter, registry
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _run_git(args: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a git command and return {"stdout": ..., "stderr": ..., "returncode": ...}.
    Never raises — all errors are returned in the dict.
    """
    cwd_path = str(Path(cwd).resolve()) if cwd else None

    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd_path,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except FileNotFoundError:
        return {"stdout": "", "stderr": "git is not installed or not on PATH.", "returncode": -1}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "git command timed out (>30s).", "returncode": -1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": -1}


def _git_output(args: List[str], cwd: Optional[str] = None) -> str:
    """Return stdout or stderr as a single string."""
    r = _run_git(args, cwd)
    if r["returncode"] == 0:
        return r["stdout"] or "(no output)"
    return r["stderr"] or f"git command failed (exit {r['returncode']})"


# ---------------------------------------------------------------------------
# git_status
# ---------------------------------------------------------------------------

@registry.register(
    name="git_status",
    description="Show the working tree status of a git repository.",
    category="git",
    parameters=[
        ToolParameter("repo_path", "string", "Path to the repository (defaults to current directory)", required=False, default="."),
    ],
    enabled=settings.tools.git_enabled,
)
def git_status(repo_path: str = ".") -> str:
    log.info("git status", extra={"repo": repo_path})
    return _git_output(["status", "--short", "--branch"], cwd=repo_path)


# ---------------------------------------------------------------------------
# git_log
# ---------------------------------------------------------------------------

@registry.register(
    name="git_log",
    description="Show recent git commit history.",
    category="git",
    parameters=[
        ToolParameter("repo_path", "string", "Path to the repository", required=False, default="."),
        ToolParameter("limit", "integer", "Number of commits to show", required=False, default=10),
    ],
    enabled=settings.tools.git_enabled,
)
def git_log(repo_path: str = ".", limit: int = 10) -> str:
    log.info("git log", extra={"repo": repo_path, "limit": limit})
    fmt = "%h %ad %an: %s"   # hash, date, author, subject
    return _git_output(
        ["log", f"--max-count={limit}", f"--pretty=format:{fmt}", "--date=short"],
        cwd=repo_path,
    )


# ---------------------------------------------------------------------------
# git_diff
# ---------------------------------------------------------------------------

@registry.register(
    name="git_diff",
    description="Show uncommitted changes in the working directory.",
    category="git",
    parameters=[
        ToolParameter("repo_path", "string", "Path to the repository", required=False, default="."),
        ToolParameter("staged", "boolean", "Show staged changes instead of unstaged", required=False, default=False),
    ],
    enabled=settings.tools.git_enabled,
)
def git_diff(repo_path: str = ".", staged: bool = False) -> str:
    log.info("git diff", extra={"repo": repo_path, "staged": staged})
    args = ["diff", "--stat"]
    if staged:
        args.append("--cached")
    return _git_output(args, cwd=repo_path)


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------

@registry.register(
    name="git_commit",
    description="Stage all changes and create a git commit with the given message.",
    category="git",
    parameters=[
        ToolParameter("message", "string", "Commit message"),
        ToolParameter("repo_path", "string", "Path to the repository", required=False, default="."),
    ],
    enabled=settings.tools.git_enabled,
)
def git_commit(message: str, repo_path: str = ".") -> str:
    if not message.strip():
        return "Commit message cannot be empty."

    log.info("git commit", extra={"message": message[:60], "repo": repo_path})

    # Stage all changes
    add_result = _run_git(["add", "-A"], cwd=repo_path)
    if add_result["returncode"] != 0:
        return f"git add failed: {add_result['stderr']}"

    commit_result = _run_git(["commit", "-m", message], cwd=repo_path)
    if commit_result["returncode"] == 0:
        return commit_result["stdout"]
    return f"git commit failed: {commit_result['stderr']}"


# ---------------------------------------------------------------------------
# git_branch
# ---------------------------------------------------------------------------

@registry.register(
    name="git_branch",
    description="List all branches, or create a new branch.",
    category="git",
    parameters=[
        ToolParameter("repo_path", "string", "Path to the repository", required=False, default="."),
        ToolParameter("create", "string", "Name of new branch to create (omit to just list branches)", required=False, default=""),
    ],
    enabled=settings.tools.git_enabled,
)
def git_branch(repo_path: str = ".", create: str = "") -> str:
    if create.strip():
        log.info("Creating git branch", extra={"branch": create, "repo": repo_path})
        return _git_output(["checkout", "-b", create.strip()], cwd=repo_path)

    log.info("Listing git branches", extra={"repo": repo_path})
    return _git_output(["branch", "-a"], cwd=repo_path)
