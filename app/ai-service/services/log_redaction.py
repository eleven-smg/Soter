"""
Structured logging helpers with guaranteed PII/secret redaction.

This module provides two things for the AI service:

1. ``RedactingFilter`` - a ``logging.Filter`` that scrubs PII-like and
   secret-like substrings out of every log record *before* it reaches a
   handler. Because it is attached at the handler level, redaction is applied
   to all log output regardless of the call site ("guaranteed redaction").

2. ``log_request`` - a helper that emits a single structured
   "request_completed" JSON log line carrying the non-sensitive request
   metadata required for observability (request id, route, method, status,
   latency, outcome, and provider).

The redaction here is intentionally lightweight and dependency-free (regex
only) so it is safe to run on the hot logging path. It complements, but is
independent of, the spaCy-based ``PIIScrubberService`` used for request
payload anonymization.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Optional

# The placeholder written in place of any redacted value.
REDACTION_PLACEHOLDER = "[REDACTED]"

# Ordered (pattern, placeholder) pairs. Applied in order, so the most specific
# / highest-risk patterns come first (e.g. cards before generic phones).
_REDACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Authorization headers / bearer tokens: "Authorization: Bearer abc..."
    (re.compile(r"(?i)\b(?:authorization|bearer|token)\b\s*[:=]?\s*[A-Za-z0-9._\-]+"),
     REDACTION_PLACEHOLDER),
    # JWTs (three base64url segments separated by dots)
    (re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
     REDACTION_PLACEHOLDER),
    # OpenAI-style API keys
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), REDACTION_PLACEHOLDER),
    # Email addresses
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
     REDACTION_PLACEHOLDER),
    # Credit-card-like sequences (13-16 digits, optional separators)
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), REDACTION_PLACEHOLDER),
    # US SSN
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), REDACTION_PLACEHOLDER),
    # International + local phone numbers
    (re.compile(r"\+?\d{1,4}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
     REDACTION_PLACEHOLDER),
    (re.compile(r"\b0\d{10}\b"), REDACTION_PLACEHOLDER),
    # Nigerian NIN (11 digits) / voter ID
    (re.compile(r"\b\d{11}\b"), REDACTION_PLACEHOLDER),
    (re.compile(r"\b[A-Z]{2}\d{8}\b"), REDACTION_PLACEHOLDER),
]

# Structured log fields known to be safe (non-sensitive); never redacted even
# if their value happens to look like a number, a path with digits, etc.
_SAFE_FIELDS = frozenset({
    "request_id", "correlationId", "route", "method", "status_code",
    "latency_ms", "outcome", "provider", "event", "levelname", "name",
    "asctime", "message",
})


def redact(value: Any) -> Any:
    """Recursively redact PII/secret-like content from ``value``.

    Strings are scrubbed with the configured patterns. Mappings and iterables
    are traversed so nested payloads are covered. Non-string scalars are
    returned unchanged.
    """
    if isinstance(value, str):
        redacted = value
        for pattern, placeholder in _REDACTION_PATTERNS:
            redacted = pattern.sub(placeholder, redacted)
        return redacted
    if isinstance(value, Mapping):
        return {k: (v if k in _SAFE_FIELDS else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return type(value)(redact(v) for v in value)
    return value


class RedactingFilter(logging.Filter):
    """Logging filter that guarantees redaction of every emitted record.

    The filter rewrites ``record.msg`` with the fully-formatted, redacted
    message and clears ``record.args`` so downstream formatters do not
    re-interpolate the original (unredacted) arguments. It also redacts any
    non-safe structured attributes attached via ``extra=...``.
    """

    #: Standard LogRecord attributes that must not be treated as user extras.
    _RESERVED = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message"}

    def filter(self, record: logging.LogRecord) -> bool:
        # 1. Redact the fully-rendered message and drop raw args.
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg = redact(message)
        record.args = ()

        # 2. Redact custom structured fields (those added via extra=...).
        for key, value in list(record.__dict__.items()):
            if key in self._RESERVED or key in _SAFE_FIELDS:
                continue
            record.__dict__[key] = redact(value)

        return True


def install_redaction(logger: Optional[logging.Logger] = None) -> None:
    """Attach a ``RedactingFilter`` to every handler on ``logger``.

    Defaults to the root logger. Idempotent: it will not add a second filter
    if one is already present on a handler.
    """
    target = logger or logging.getLogger()
    for handler in target.handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())


def log_request(
    logger: logging.Logger,
    *,
    request_id: str,
    method: str,
    route: str,
    status_code: int,
    latency_ms: float,
    outcome: str,
    provider: Optional[str] = None,
    **extra: Any,
) -> None:
    """Emit one structured JSON log line describing a completed request.

    All first-class fields are non-sensitive by design. ``latency_ms`` is
    rounded for stable, low-cardinality output. Any additional ``extra``
    fields still pass through the RedactingFilter.
    """
    payload: dict[str, Any] = {
        "event": "request_completed",
        "request_id": request_id,
        "method": method,
        "route": route,
        "status_code": status_code,
        "latency_ms": round(float(latency_ms), 2),
        "outcome": outcome,
        "provider": provider,
    }
    if extra:
        payload.update(extra)
    logger.info("request_completed", extra=payload)
