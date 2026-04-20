"""
Email tool — send emails via SMTP.

Credentials are loaded from environment variables (never from config.yaml):
    JARVIS_EMAIL_USER=you@gmail.com
    JARVIS_EMAIL_PASSWORD=your-app-password

Gmail users: create an App Password at https://myaccount.google.com/apppasswords
"""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

from config import settings
from tools.registry import ToolParameter, registry
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# send_email
# ---------------------------------------------------------------------------

@registry.register(
    name="send_email",
    description=(
        "Send an email to one or more recipients. "
        "Requires JARVIS_EMAIL_USER and JARVIS_EMAIL_PASSWORD environment variables."
    ),
    category="email",
    parameters=[
        ToolParameter("to", "string", "Recipient email address (or comma-separated list)"),
        ToolParameter("subject", "string", "Email subject line"),
        ToolParameter("body", "string", "Email body text"),
        ToolParameter("cc", "string", "CC recipients (comma-separated)", required=False, default=""),
        ToolParameter("html", "boolean", "Send as HTML email instead of plain text", required=False, default=False),
    ],
    enabled=True,   # always registered; runtime gate is the check inside the function
)
def send_email(
    to: str,
    subject: str,
    body: str,
    cc: str = "",
    html: bool = False,
) -> str:
    if not settings.tools.email.enabled:
        return "Email tool is disabled. Set tools.email.enabled=true in config.yaml."

    sender = os.environ.get("JARVIS_EMAIL_USER", "")
    password = os.environ.get("JARVIS_EMAIL_PASSWORD", "")

    if not sender or not password:
        return (
            "Email credentials not configured. "
            "Set JARVIS_EMAIL_USER and JARVIS_EMAIL_PASSWORD in your .env file."
        )

    recipients = [r.strip() for r in to.split(",") if r.strip()]
    cc_list = [r.strip() for r in cc.split(",") if r.strip()]

    if not recipients:
        return "No valid recipient email address provided."

    log.info(
        "Sending email",
        extra={"to": recipients, "subject": subject[:60], "html": html},
    )

    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)

    mime_type = "html" if html else "plain"
    msg.attach(MIMEText(body, mime_type))

    all_recipients = recipients + cc_list

    try:
        with smtplib.SMTP(settings.tools.email.smtp_host, settings.tools.email.smtp_port) as server:
            if settings.tools.email.use_tls:
                server.starttls()
            server.login(sender, password)
            server.sendmail(sender, all_recipients, msg.as_string())

        log.info("Email sent", extra={"to": recipients, "subject": subject[:60]})
        return f"Email sent successfully to {', '.join(recipients)}."

    except smtplib.SMTPAuthenticationError:
        return "Authentication failed. Check your JARVIS_EMAIL_USER and JARVIS_EMAIL_PASSWORD."
    except smtplib.SMTPConnectError:
        return f"Could not connect to SMTP server {settings.tools.email.smtp_host}:{settings.tools.email.smtp_port}."
    except Exception as exc:
        log.error("Email send failed", extra={"error": str(exc)})
        return f"Failed to send email: {exc}"
