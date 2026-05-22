"""
Email notification utilities for dashboard-triggered predictions.
"""

import os
import re
import smtplib

from datetime import datetime
from email.message import EmailMessage


EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def parse_recipients(raw):
    """
    Parse comma- or semicolon-separated recipient emails.
    """

    if not raw:
        return []

    recipients = [
        part.strip()
        for part in re.split(r"[,;]", str(raw))
        if part.strip()
    ]
    invalid = [email for email in recipients if not EMAIL_PATTERN.match(email)]

    if invalid:
        raise ValueError(f"Invalid recipient email address(es): {', '.join(invalid)}")

    return recipients


def _format_probability(value):
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def _format_decimal(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def build_prediction_email(prediction, ticker, recipients, from_email):
    """
    Build a plain-text prediction notification email.
    """

    msg = EmailMessage()
    direction = prediction.get("direction", "UNKNOWN")
    confidence = prediction.get("confidence", "UNKNOWN")

    msg["Subject"] = f"QUSA prediction: {ticker} {direction} ({confidence})"
    msg["From"] = from_email
    msg["To"] = ", ".join(recipients)

    body = "\n".join(
        [
            f"QUSA prediction generated for {ticker}.",
            "",
            "Latest prediction:",
            f"- Date: {prediction.get('date', 'Unknown')}",
            f"- Direction: {direction}",
            f"- Probability up: {_format_probability(prediction.get('probability_up'))}",
            f"- Confidence: {confidence}",
            f"- ATR%: {_format_decimal(prediction.get('atr_pct'))}",
            f"- Volatility filter triggered: {prediction.get('volatility_filter_triggered', 'Unknown')}",
            f"- Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "This is an automated research notification, not financial advice.",
        ]
    )
    msg.set_content(body)

    return msg


def _resolve_smtp_settings(email_config):
    if not email_config.get("enabled", False):
        raise ValueError("Email notifications are disabled in config.")

    smtp_host = email_config.get("smtp_host")
    smtp_port = email_config.get("smtp_port")
    smtp_user_env = email_config.get("smtp_user_env", "QUSA_SMTP_USER")
    smtp_password_env = email_config.get("smtp_password_env", "QUSA_SMTP_PASSWORD")
    smtp_user = os.getenv(smtp_user_env)
    smtp_password = os.getenv(smtp_password_env)

    missing = []
    if not smtp_host:
        missing.append("smtp_host")
    if not smtp_port:
        missing.append("smtp_port")
    if not smtp_user:
        missing.append(smtp_user_env)
    if not smtp_password:
        missing.append(smtp_password_env)

    if missing:
        raise ValueError(f"Missing email configuration: {', '.join(missing)}")

    return {
        "smtp_host": smtp_host,
        "smtp_port": int(smtp_port),
        "smtp_user": smtp_user,
        "smtp_password": smtp_password,
        "from_email": email_config.get("from_email") or smtp_user,
        "use_tls": email_config.get("use_tls", True),
    }


def send_prediction_email(email_config, recipients, prediction, ticker):
    """
    Send a prediction notification email.
    """

    result = {
        "sent": False,
        "recipients": recipients,
        "error": None,
    }

    try:
        if not recipients:
            raise ValueError("At least one recipient email address is required.")

        settings = _resolve_smtp_settings(email_config or {})
        msg = build_prediction_email(
            prediction=prediction,
            ticker=ticker,
            recipients=recipients,
            from_email=settings["from_email"],
        )

        with smtplib.SMTP(
            settings["smtp_host"],
            settings["smtp_port"],
            timeout=30,
        ) as server:
            if settings["use_tls"]:
                server.starttls()
            server.login(settings["smtp_user"], settings["smtp_password"])
            server.send_message(msg)

        result["sent"] = True
        return result

    except Exception as exc:
        result["error"] = str(exc)
        return result
