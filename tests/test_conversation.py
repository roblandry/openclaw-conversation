"""Tests for OpenClaw conversation response handling."""

from __future__ import annotations

import pytest

from custom_components.openclaw_conversation.conversation import (
    GENERIC_ERROR_RESPONSE,
    NETWORK_ERROR_RESPONSE,
    SUPPORTED_LANGUAGES,
    TIMEOUT_RESPONSE,
    OpenClawConversationAgent,
    _parse_openclaw_response,
)


def test_parse_streaming_response() -> None:
    """Parse assistant deltas from a server-sent event response."""
    raw = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" there"}}]}',
            "data: [DONE]",
        ]
    )

    assert _parse_openclaw_response(raw) == "Hello there"


def test_parse_json_response() -> None:
    """Parse assistant text from a non-streaming response."""
    raw = '{"choices":[{"message":{"content":"Plain response"}}]}'

    assert _parse_openclaw_response(raw) == "Plain response"


def test_parse_error_response() -> None:
    """Raise a readable error when OpenClaw returns an error payload."""
    raw = '{"error":{"message":"model unavailable"}}'

    with pytest.raises(RuntimeError, match="model unavailable"):
        _parse_openclaw_response(raw)


def test_parse_empty_stream_response() -> None:
    """Raise a specific error when a stream completes without content."""
    with pytest.raises(RuntimeError, match="empty stream"):
        _parse_openclaw_response("data: [DONE]")


def test_parse_stream_error_code() -> None:
    """Use an error code when a stream error has no message."""
    raw = 'data: {"error":{"code":"gateway_timeout"}}'

    with pytest.raises(RuntimeError, match="gateway_timeout"):
        _parse_openclaw_response(raw)


def test_parse_stream_error_object_without_message() -> None:
    """Serialize structured stream errors that lack a message or code."""
    raw = 'data: {"error":{"detail":"bad"}}'

    with pytest.raises(RuntimeError, match='"detail": "bad"'):
        _parse_openclaw_response(raw)


def test_parse_stream_error_scalar() -> None:
    """Use scalar stream errors directly."""
    raw = 'data: {"error":"bad"}'

    with pytest.raises(RuntimeError, match="bad"):
        _parse_openclaw_response(raw)


def test_parse_ignores_malformed_stream_chunks() -> None:
    """Skip malformed and irrelevant stream chunks while collecting content."""
    raw = "\n".join(
        [
            "event: message",
            "data: not-json",
            'data: {"choices":[]}',
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
        ]
    )

    assert _parse_openclaw_response(raw) == "ok"


@pytest.mark.parametrize(
    "raw",
    [
        "data: []",
        'data: {"choices":[123]}',
        'data: {"choices":[{"delta":"bad"}]}',
    ],
)
def test_parse_ignores_malformed_stream_shapes(raw: str) -> None:
    """Ignore malformed stream JSON shapes."""
    with pytest.raises(RuntimeError, match="No response from OpenClaw"):
        _parse_openclaw_response(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        '{"choices":[]}',
        '{"choices":[123]}',
        '{"choices":[{"message":"bad"}]}',
    ],
)
def test_parse_ignores_malformed_json_shapes(raw: str) -> None:
    """Ignore malformed non-streaming JSON shapes."""
    with pytest.raises(RuntimeError, match="No response from OpenClaw"):
        _parse_openclaw_response(raw)


def test_parse_missing_content_response() -> None:
    """Raise a readable error for responses without assistant content."""
    with pytest.raises(RuntimeError, match="No response from OpenClaw"):
        _parse_openclaw_response('{"choices":[{"message":{}}]}')


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        ("bad", 0),
        (-10, 0),
        ("15", 15),
    ],
)
def test_normalize_timeout(value: object, expected: int) -> None:
    """Normalize timeout values to non-negative integers."""
    assert OpenClawConversationAgent._normalize_timeout(value) == expected


def test_supported_languages_are_english_only() -> None:
    """Advertise only languages with English fallback speech."""
    assert SUPPORTED_LANGUAGES == ["en"]


def test_english_fallback_responses_are_centralized() -> None:
    """Keep user-facing fallback speech strings centralized."""
    assert TIMEOUT_RESPONSE == "OpenClaw took too long to respond."
    assert NETWORK_ERROR_RESPONSE == "Network error while contacting OpenClaw."
    assert GENERIC_ERROR_RESPONSE == "Error communicating with OpenClaw."
