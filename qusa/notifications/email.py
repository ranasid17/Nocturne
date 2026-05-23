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


def _get_direction_color(direction):
    direction = str(direction).upper()
    if direction == "UP":
        return "#10b981"  # Green
    if direction == "DOWN":
        return "#ef4444"  # Red
    return "#64748b"  # Gray


def _build_html_body(prediction, ticker):
    direction = prediction.get("direction", "UNKNOWN")
    confidence = prediction.get("confidence", "UNKNOWN")
    color = _get_direction_color(direction)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f8fafc;
            margin: 0;
            padding: 20px;
            color: #1e293b;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid #e2e8f0;
        }}
        .header {{
            background-color: #2563eb;
            color: #ffffff;
            padding: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 20px;
            font-weight: 700;
        }}
        .content {{
            padding: 30px;
        }}
        .ticker {{
            font-size: 14px;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .signal-card {{
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            background-color: #f1f5f9;
            margin-bottom: 24px;
        }}
        .direction {{
            font-size: 36px;
            font-weight: 800;
            color: {color};
            margin: 0;
        }}
        .confidence {{
            font-size: 14px;
            font-weight: 600;
            color: #64748b;
            margin-top: 4px;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .metrics-table td {{
            padding: 12px 0;
            border-bottom: 1px solid #f1f5f9;
        }}
        .metric-label {{
            font-size: 13px;
            font-weight: 600;
            color: #64748b;
        }}
        .metric-value {{
            font-size: 13px;
            font-weight: 700;
            color: #1e293b;
            text-align: right;
        }}
        .footer {{
            padding: 20px;
            text-align: center;
            font-size: 12px;
            color: #94a3b8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Nocturne Intelligence</h1>
        </div>
        <div class="content">
            <div class="ticker">{ticker} Analysis</div>
            <div class="signal-card">
                <div class="direction">{direction}</div>
                <div class="confidence">{confidence} CONVICTION</div>
            </div>
            
            <table class="metrics-table">
                <tr>
                    <td class="metric-label">Prediction Date</td>
                    <td class="metric-value">{prediction.get('date', 'Unknown')}</td>
                </tr>
                <tr>
                    <td class="metric-label">Probability Up</td>
                    <td class="metric-value">{_format_probability(prediction.get('probability_up'))}</td>
                </tr>
                <tr>
                    <td class="metric-label">ATR%</td>
                    <td class="metric-value">{_format_decimal(prediction.get('atr_pct'))}</td>
                </tr>
                <tr>
                    <td class="metric-label">Volatility Filter</td>
                    <td class="metric-value">{'TRIGGERED' if prediction.get('volatility_filter_triggered') else 'CLEAR'}</td>
                </tr>
                <tr>
                    <td class="metric-label">Generated At</td>
                    <td class="metric-value">{timestamp}</td>
                </tr>
            </table>
        </div>
        <div class="footer">
            This is an automated research notification, not financial advice.
        </div>
    </div>
</body>
</html>
"""


def build_prediction_email(prediction, ticker, recipients, from_email):
    """
    Build a multi-part prediction notification email (Plain Text + HTML).
    """

    msg = EmailMessage()
    direction = prediction.get("direction", "UNKNOWN")
    confidence = prediction.get("confidence", "UNKNOWN")

    msg["Subject"] = f"QUSA prediction: {ticker} {direction} ({confidence})"
    msg["From"] = from_email
    msg["To"] = ", ".join(recipients)

    # Plain text version
    text_body = "\n".join(
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
    msg.set_content(text_body)

    # HTML version
    html_body = _build_html_body(prediction, ticker)
    msg.add_alternative(html_body, subtype="html")

    return msg


def _resolve_smtp_settings(email_config):
    if not email_config.get("enabled", False):
        raise ValueError("Email notifications are disabled in config.")

    smtp_host = email_config.get("smtp_host")
    smtp_port = email_config.get("smtp_port")
    smtp_user_env = email_config.get("smtp_user_env", "QUSA_SMTP_USER")
    smtp_password_env = email_config.get("smtp_password_env", "QUSA_SMTP_PASSWORD")
    smtp_user = email_config.get("smtp_user") or os.getenv(smtp_user_env)
    smtp_password = email_config.get("smtp_password") or os.getenv(smtp_password_env)

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
