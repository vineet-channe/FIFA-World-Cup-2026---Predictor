BRIER_ALERT_THRESHOLD = 0.22
BRIER_ROLLBACK_THRESHOLD = 0.24


def should_deploy(brier: float | None) -> tuple[bool, str]:
    """
    Returns (deploy: bool, reason: str).
    None brier = fewer than 5 WC matches available = always deploy.
    """
    if brier is None:
        return True, "no validation data yet — deployed unconditionally"
    if brier > BRIER_ROLLBACK_THRESHOLD:
        return False, f"Brier {brier:.4f} exceeds rollback threshold {BRIER_ROLLBACK_THRESHOLD}"
    if brier > BRIER_ALERT_THRESHOLD:
        return True, f"WARNING: Brier {brier:.4f} above alert threshold — deployed but monitor"
    return True, f"Brier {brier:.4f} — healthy"
