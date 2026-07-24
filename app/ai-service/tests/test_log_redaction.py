"""Tests for guaranteed log redaction and structured request logging (#461)."""

import json
import logging
from io import StringIO

import pytest
from pythonjsonlogger import jsonlogger

from services.log_redaction import (
    REDACTION_PLACEHOLDER,
    RedactingFilter,
    install_redaction,
    log_request,
    redact,
)

# Representative PII / secret strings that must never survive into logs.
PII_SAMPLES = [
    "jane.doe@example.com",
    "support@pulsefy.org",
    "+234 803 123 4567",
    "08029876543",
    "12345678901",                    # Nigerian NIN
    "4111 1111 1111 1111",            # credit card
    "123-45-6789",                    # US SSN
    "sk-abcdEFGH1234567890abcdEFGH",  # API key
    "eyJhbGciOiJIUzI1NiJ.eyJzdWIiOiIxMjM0.abcDEF123456",  # JWT
]


def _make_capturing_logger(name: str):
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        jsonlogger.JsonFormatter("%(levelname)s %(name)s %(message)s")
    )
    logger = logging.getLogger(name)
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    install_redaction(logger)
    return logger, stream


class TestRedactFunction:
    @pytest.mark.parametrize("sample", PII_SAMPLES)
    def test_redact_removes_pii(self, sample):
        text = f"user supplied value: {sample} end"
        result = redact(text)
        assert sample not in result
        assert REDACTION_PLACEHOLDER in result

    def test_redact_traverses_nested_structures(self):
        payload = {
            "email": "jane.doe@example.com",
            "nested": ["call +234 803 123 4567"],
        }
        dumped = json.dumps(redact(payload))
        assert "jane.doe@example.com" not in dumped
        assert "+234 803 123 4567" not in dumped

    def test_redact_preserves_non_pii(self):
        text = "outcome=success route=/v1/ai/ocr"
        assert redact(text) == text


class TestRedactingFilter:
    @pytest.mark.parametrize("sample", PII_SAMPLES)
    def test_pii_never_appears_in_logs(self, sample):
        logger, stream = _make_capturing_logger(f"redact.msg.{abs(hash(sample))}")
        logger.info("processing payload=%s for request", sample)
        output = stream.getvalue()
        assert sample not in output
        assert REDACTION_PLACEHOLDER in output

    def test_pii_in_extra_is_redacted(self):
        logger, stream = _make_capturing_logger("redact.extra")
        logger.info("payload received", extra={"body": "email jane.doe@example.com"})
        output = stream.getvalue()
        assert "jane.doe@example.com" not in output
        assert REDACTION_PLACEHOLDER in output

    def test_filter_is_idempotent(self):
        logger, _ = _make_capturing_logger("redact.idempotent")
        install_redaction(logger)
        install_redaction(logger)
        filters = [
            f
            for h in logger.handlers
            for f in h.filters
            if isinstance(f, RedactingFilter)
        ]
        assert len(filters) == 1


class TestLogRequest:
    def test_structured_fields_present(self):
        logger, stream = _make_capturing_logger("redact.request")
        log_request(
            logger,
            request_id="req-123",
            method="POST",
            route="/v1/ai/anonymize",
            status_code=200,
            latency_ms=12.3456,
            outcome="success",
            provider="openai",
        )
        record = json.loads(stream.getvalue().strip().splitlines()[-1])
        assert record["event"] == "request_completed"
        assert record["request_id"] == "req-123"
        assert record["method"] == "POST"
        assert record["route"] == "/v1/ai/anonymize"
        assert record["status_code"] == 200
        assert record["latency_ms"] == 12.35
        assert record["outcome"] == "success"
        assert record["provider"] == "openai"

    def test_request_log_redacts_extra_pii(self):
        logger, stream = _make_capturing_logger("redact.request.pii")
        log_request(
            logger,
            request_id="req-123",
            method="POST",
            route="/v1/ai/anonymize",
            status_code=200,
            latency_ms=5.0,
            outcome="success",
            provider="openai",
            note="contact jane.doe@example.com",
        )
        output = stream.getvalue()
        assert "jane.doe@example.com" not in output
        record = json.loads(output.strip().splitlines()[-1])
        assert record["route"] == "/v1/ai/anonymize"  # safe field survives
