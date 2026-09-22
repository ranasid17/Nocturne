"""Explicit worker helpers for notification delivery."""

from .email import send_prediction_email


def deliver_next_notification(repository, email_config):
    """Deliver one claimed message. HTTP read routes must never call this."""

    attempt = repository.claim_notification()
    if attempt is None:
        return None
    payload = attempt["payload"]
    result = send_prediction_email(
        email_config,
        attempt["recipients"],
        payload.get("prediction", payload),
        payload.get("ticker", "UNKNOWN"),
    )
    repository.complete_notification(
        attempt["id"], "sent" if result["sent"] else "failed", result["error"]
    )
    return {"id": attempt["id"], **result}
