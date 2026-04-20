"""
Tests for the Tools Layer (system, file, git, email).
No real file system mutations are made — temp dirs are used for file tests.
No real git operations run against the live repo.
No real emails are sent — SMTP is mocked.

Run:
    pytest tests/test_tools.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force tool registration before tests run
from tools.loader import load_all_tools
load_all_tools()

from tools.registry import registry


# ---------------------------------------------------------------------------
# Tool Registry — confirm all tools registered
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_system_tools_registered(self):
        names = [t.name for t in registry.list_tools()]
        assert "open_app" in names
        assert "get_system_info" in names
        assert "list_processes" in names
        assert "kill_process" in names

    def test_file_tools_registered(self):
        names = [t.name for t in registry.list_tools()]
        assert "read_file" in names
        assert "write_file" in names
        assert "list_directory" in names
        assert "search_files" in names
        assert "delete_file" in names

    def test_git_tools_registered(self):
        names = [t.name for t in registry.list_tools()]
        assert "git_status" in names
        assert "git_log" in names
        assert "git_diff" in names
        assert "git_commit" in names
        assert "git_branch" in names

    def test_email_tool_registered(self):
        names = [t.name for t in registry.list_tools()]
        assert "send_email" in names

    def test_all_tools_have_schemas(self):
        for tool in registry.list_tools():
            schema = tool.to_schema()
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema


# ---------------------------------------------------------------------------
# System Tools
# ---------------------------------------------------------------------------

class TestSystemTools:
    def test_get_system_info_returns_dict(self):
        result = registry.execute("get_system_info")
        assert isinstance(result, dict)
        assert "cpu_percent" in result
        assert "ram_percent" in result
        assert "os" in result

    def test_list_processes_returns_list(self):
        result = registry.execute("list_processes")
        assert isinstance(result, list)
        assert len(result) > 0
        assert "pid" in result[0]
        assert "name" in result[0]

    def test_list_processes_filter(self):
        result = registry.execute("list_processes", filter_name="python")
        assert isinstance(result, list)
        # At minimum the current python process should appear
        names = [p["name"].lower() for p in result]
        assert any("python" in n for n in names)

    def test_kill_process_blocked_without_permission(self):
        result = registry.execute("kill_process", identifier="99999999")
        assert "disabled" in result.lower() or "no process" in result.lower()

    def test_open_app_nonexistent_returns_error(self):
        result = registry.execute("open_app", app_name="__nonexistent_app_xyz__")
        assert isinstance(result, str)
        assert len(result) > 0   # some error message


# ---------------------------------------------------------------------------
# File Tools
# ---------------------------------------------------------------------------

class TestFileTools:
    def test_write_and_read_file(self, tmp_path):
        file_path = str(tmp_path / "test.txt")
        write_result = registry.execute("write_file", path=file_path, content="Hello JARVIS!")
        assert "Wrote" in write_result

        read_result = registry.execute("read_file", path=file_path)
        assert "Hello JARVIS!" in read_result

    def test_append_to_file(self, tmp_path):
        file_path = str(tmp_path / "append.txt")
        registry.execute("write_file", path=file_path, content="Line 1\n")
        registry.execute("write_file", path=file_path, content="Line 2\n", append=True)
        content = registry.execute("read_file", path=file_path)
        assert "Line 1" in content
        assert "Line 2" in content

    def test_read_nonexistent_file(self, tmp_path):
        result = registry.execute("read_file", path=str(tmp_path / "ghost.txt"))
        assert "not found" in result.lower()

    def test_read_file_max_lines(self, tmp_path):
        file_path = str(tmp_path / "big.txt")
        lines = "\n".join(f"line {i}" for i in range(500))
        Path(file_path).write_text(lines)
        result = registry.execute("read_file", path=file_path, max_lines=10)
        assert "truncated" in result
        assert result.count("\n") < 20

    def test_list_directory(self, tmp_path):
        (tmp_path / "file_a.txt").write_text("a")
        (tmp_path / "file_b.txt").write_text("b")
        (tmp_path / "subdir").mkdir()

        result = registry.execute("list_directory", path=str(tmp_path))
        names = [item["name"] for item in result]
        assert "file_a.txt" in names
        assert "file_b.txt" in names
        assert "subdir" in names

    def test_list_directory_hidden_files(self, tmp_path):
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "visible.txt").write_text("")

        result_no_hidden = registry.execute("list_directory", path=str(tmp_path), show_hidden=False)
        result_with_hidden = registry.execute("list_directory", path=str(tmp_path), show_hidden=True)

        names_no_hidden = [i["name"] for i in result_no_hidden]
        names_with_hidden = [i["name"] for i in result_with_hidden]
        assert ".hidden" not in names_no_hidden
        assert ".hidden" in names_with_hidden

    def test_search_files(self, tmp_path):
        (tmp_path / "alpha.py").write_text("")
        (tmp_path / "beta.py").write_text("")
        (tmp_path / "config.yaml").write_text("")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "gamma.py").write_text("")

        results = registry.execute("search_files", pattern="*.py", root=str(tmp_path))
        assert len(results) == 3
        assert all(r.endswith(".py") for r in results)

    def test_delete_file_blocked_without_permission(self, tmp_path):
        file_path = str(tmp_path / "to_delete.txt")
        Path(file_path).write_text("content")
        result = registry.execute("delete_file", path=file_path)
        # Should be blocked (allow_file_delete=false in default config)
        assert "disabled" in result.lower()
        assert Path(file_path).exists()   # file still there

    def test_delete_file_with_permission(self, tmp_path):
        from config import settings
        file_path = str(tmp_path / "to_delete.txt")
        Path(file_path).write_text("content")

        original = settings.tools.system.allow_file_delete
        settings.tools.system.allow_file_delete = True
        try:
            result = registry.execute("delete_file", path=file_path)
            assert "Deleted" in result
            assert not Path(file_path).exists()
        finally:
            settings.tools.system.allow_file_delete = original


# ---------------------------------------------------------------------------
# Git Tools
# ---------------------------------------------------------------------------

class TestGitTools:
    def _mock_git(self, stdout="ok", returncode=0):
        return {"stdout": stdout, "stderr": "", "returncode": returncode}

    def test_git_status(self):
        with patch("tools.git_tools._run_git", return_value=self._mock_git("## main\nM  file.py")):
            result = registry.execute("git_status")
        assert "main" in result or "file.py" in result

    def test_git_log(self):
        log_output = "abc1234 2024-01-01 Alice: Initial commit"
        with patch("tools.git_tools._run_git", return_value=self._mock_git(log_output)):
            result = registry.execute("git_log", limit=5)
        assert "Initial commit" in result

    def test_git_diff(self):
        with patch("tools.git_tools._run_git", return_value=self._mock_git("1 file changed")):
            result = registry.execute("git_diff")
        assert "file changed" in result

    def test_git_commit_empty_message(self):
        result = registry.execute("git_commit", message="")
        assert "empty" in result.lower()

    def test_git_commit_success(self):
        with patch("tools.git_tools._run_git", return_value=self._mock_git("[main abc1234] My commit")):
            result = registry.execute("git_commit", message="My commit")
        assert "commit" in result.lower() or "main" in result

    def test_git_branch_list(self):
        with patch("tools.git_tools._run_git", return_value=self._mock_git("* main\n  dev")):
            result = registry.execute("git_branch")
        assert "main" in result

    def test_git_branch_create(self):
        with patch("tools.git_tools._run_git", return_value=self._mock_git("Switched to a new branch 'feature-x'")):
            result = registry.execute("git_branch", create="feature-x")
        assert "feature-x" in result

    def test_git_not_installed(self):
        with patch("tools.git_tools._run_git", return_value={"stdout": "", "stderr": "git is not installed or not on PATH.", "returncode": -1}):
            result = registry.execute("git_status")
        assert "git" in result.lower()


# ---------------------------------------------------------------------------
# Email Tool
# ---------------------------------------------------------------------------

class TestEmailTool:
    def test_email_disabled_by_default(self):
        # email.enabled=False in default config — function returns a message, not an exception
        from config import settings
        assert not settings.tools.email.enabled   # sanity-check default
        result = registry.execute(
            "send_email", to="test@example.com", subject="Hi", body="Hello"
        )
        assert "disabled" in result.lower()

    def test_email_missing_credentials(self):
        from config import settings
        original = settings.tools.email.enabled
        settings.tools.email.enabled = True
        try:
            with patch.dict(os.environ, {"JARVIS_EMAIL_USER": "", "JARVIS_EMAIL_PASSWORD": ""}):
                result = registry.execute(
                    "send_email", to="test@example.com", subject="Hi", body="Hello"
                )
            assert "credentials" in result.lower() or "not configured" in result.lower()
        finally:
            settings.tools.email.enabled = original

    def test_email_empty_recipient(self):
        from config import settings
        original = settings.tools.email.enabled
        settings.tools.email.enabled = True
        try:
            with patch.dict(os.environ, {"JARVIS_EMAIL_USER": "a@b.com", "JARVIS_EMAIL_PASSWORD": "pw"}):
                result = registry.execute(
                    "send_email", to="  ", subject="Hi", body="Hello"
                )
            assert "recipient" in result.lower() or "valid" in result.lower()
        finally:
            settings.tools.email.enabled = original

    def test_email_sends_when_configured(self):
        from config import settings
        original_enabled = settings.tools.email.enabled
        original_host = settings.tools.email.smtp_host
        settings.tools.email.enabled = True
        settings.tools.email.smtp_host = "smtp.gmail.com"

        try:
            with patch.dict(os.environ, {
                "JARVIS_EMAIL_USER": "sender@gmail.com",
                "JARVIS_EMAIL_PASSWORD": "app-password",
            }):
                with patch("smtplib.SMTP") as mock_smtp:
                    mock_server = MagicMock()
                    mock_smtp.return_value.__enter__.return_value = mock_server

                    result = registry.execute(
                        "send_email",
                        to="recipient@example.com",
                        subject="Test",
                        body="Hello from JARVIS",
                    )

            assert "sent successfully" in result.lower()
            mock_server.sendmail.assert_called_once()
        finally:
            settings.tools.email.enabled = original_enabled
            settings.tools.email.smtp_host = original_host


# ---------------------------------------------------------------------------
# Tool Loader
# ---------------------------------------------------------------------------

class TestToolLoader:
    def test_load_all_tools_returns_count(self):
        from tools.loader import load_all_tools
        count = load_all_tools()
        assert count >= 13    # at least 4 system + 5 file + 5 git + 1 email
