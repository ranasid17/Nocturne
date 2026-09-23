"""Safe diagnostics for exceptions from providers and local configuration."""


def safe_error(error):
    """Categorize failures without retaining arbitrary, potentially secret text."""
    known = {
        "Required artifact was not found.",
        "Access to a required resource was denied.",
        "Input or configuration was invalid.",
        "An operation timed out.",
        "An external service was unavailable.",
        "Operation failed; check configuration and required resources.",
    }
    if isinstance(error, str):
        return error if error in known else "Operation failed; check configuration and required resources."
    if isinstance(error, FileNotFoundError):
        return "Required artifact was not found."
    if isinstance(error, PermissionError):
        return "Access to a required resource was denied."
    if isinstance(error, (ValueError, TypeError)):
        return "Input or configuration was invalid."
    if isinstance(error, TimeoutError):
        return "An operation timed out."
    if isinstance(error, ConnectionError):
        return "An external service was unavailable."
    return "Operation failed; check configuration and required resources."
