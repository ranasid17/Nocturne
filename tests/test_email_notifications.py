from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qusa.notifications import (
    build_prediction_email,
    parse_recipients,
    send_prediction_email,
)


def _email_config():
    return {
        "enabled": True,
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "use_tls": True,
        "smtp_user_env": "QUSA_SMTP_USER",
        "smtp_password_env": "QUSA_SMTP_PASSWORD",
        "from_email": "",
    }


def _prediction():
    return {
        "date": "2026-05-21",
        "direction": "UP",
        "probability_up": 0.73,
        "confidence": "HIGH",
        "atr_pct": 1.25,
        "volatility_filter_triggered": False,
    }


def test_parse_recipients_accepts_commas_and_semicolons():
    recipients = parse_recipients("one@example.com, two@example.com;three@example.com")

    assert recipients == [
        "one@example.com",
        "two@example.com",
        "three@example.com",
    ]


def test_parse_recipients_rejects_invalid_addresses():
    with pytest.raises(ValueError, match="Invalid recipient"):
        parse_recipients("valid@example.com, invalid-address")


def test_build_prediction_email_contains_html_alternative():
    msg = build_prediction_email(
        prediction=_prediction(),
        ticker="UPRO",
        recipients=["desk@example.com"],
        from_email="sender@example.com",
    )

    # EmailMessage with alternatives should have multiple parts
    parts = list(msg.iter_parts())
    assert len(parts) == 2  # text/plain and text/html

    # Check text part (first part)
    text_part = parts[0].get_content()
    assert "UPRO" in text_part
    assert "Probability up: 73.0%" in text_part

    # Check HTML part (second part)
    html_part = parts[1].get_content()
    assert "<!DOCTYPE html>" in html_part
    assert "Nocturne Intelligence" in html_part
    assert "UPRO Analysis" in html_part
    assert "UP" in html_part
    assert "HIGH CONVICTION" in html_part
    assert "73.0%" in html_part


def test_send_prediction_email_uses_env_credentials_and_default_sender(monkeypatch):
    monkeypatch.setenv("QUSA_SMTP_USER", "smtp-user@example.com")
    monkeypatch.setenv("QUSA_SMTP_PASSWORD", "secret")
    smtp_server = MagicMock()
    smtp_context = MagicMock()
    smtp_context.__enter__.return_value = smtp_server
    smtp_context.__exit__.return_value = None

    with patch("qusa.notifications.email.smtplib.SMTP", return_value=smtp_context) as smtp:
        result = send_prediction_email(
            email_config=_email_config(),
            recipients=["desk@example.com", "pm@example.com"],
            prediction=_prediction(),
            ticker="UPRO",
        )

    assert result == {
        "sent": True,
        "recipients": ["desk@example.com", "pm@example.com"],
        "error": None,
    }
    smtp.assert_called_once_with("smtp.example.com", 587, timeout=30)
    smtp_server.starttls.assert_called_once()
    smtp_server.login.assert_called_once_with("smtp-user@example.com", "secret")
    sent_msg = smtp_server.send_message.call_args.args[0]
    assert sent_msg["From"] == "smtp-user@example.com"
    assert sent_msg["To"] == "desk@example.com, pm@example.com"


def test_send_prediction_email_accepts_runtime_credentials(monkeypatch):
    monkeypatch.delenv("QUSA_SMTP_USER", raising=False)
    monkeypatch.delenv("QUSA_SMTP_PASSWORD", raising=False)
    config = _email_config()
    config["smtp_user"] = "runtime-user@example.com"
    config["smtp_password"] = "runtime-secret"
    smtp_server = MagicMock()
    smtp_context = MagicMock()
    smtp_context.__enter__.return_value = smtp_server
    smtp_context.__exit__.return_value = None

    with patch("qusa.notifications.email.smtplib.SMTP", return_value=smtp_context):
        result = send_prediction_email(
            email_config=config,
            recipients=["desk@example.com"],
            prediction=_prediction(),
            ticker="UPRO",
        )

    assert result["sent"] is True
    smtp_server.login.assert_called_once_with(
        "runtime-user@example.com",
        "runtime-secret",
    )


def test_send_prediction_email_returns_error_on_smtp_failure(monkeypatch):
    monkeypatch.setenv("QUSA_SMTP_USER", "smtp-user@example.com")
    monkeypatch.setenv("QUSA_SMTP_PASSWORD", "secret")

    with patch("qusa.notifications.email.smtplib.SMTP", side_effect=OSError("network down")):
        result = send_prediction_email(
            email_config=_email_config(),
            recipients=["desk@example.com"],
            prediction=_prediction(),
            ticker="UPRO",
        )

    assert result["sent"] is False
    assert result["recipients"] == ["desk@example.com"]
    assert "network down" in result["error"]


def test_send_prediction_email_reports_missing_env(monkeypatch):
    monkeypatch.delenv("QUSA_SMTP_USER", raising=False)
    monkeypatch.delenv("QUSA_SMTP_PASSWORD", raising=False)

    result = send_prediction_email(
        email_config=_email_config(),
        recipients=["desk@example.com"],
        prediction=_prediction(),
        ticker="UPRO",
    )

    assert result["sent"] is False
    assert "QUSA_SMTP_USER" in result["error"]
    assert "QUSA_SMTP_PASSWORD" in result["error"]
