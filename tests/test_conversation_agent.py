"""Functional tests for the OpenClaw conversation agent."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import aiohttp
import pytest
from homeassistant.components.conversation.models import (
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Context, HomeAssistant

import custom_components.openclaw_conversation.conversation as conversation_module
from custom_components.openclaw_conversation.const import (
    CONF_AGENT_ID,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_SESSION_KEY,
    CONF_STRIP_EMOJI,
    CONF_SYSTEM_PROMPT,
    CONF_TIMEOUT,
    DEFAULT_AGENT_ID,
    DEFAULT_SESSION_KEY,
)
from custom_components.openclaw_conversation.conversation import (
    GENERIC_ERROR_RESPONSE,
    NETWORK_ERROR_RESPONSE,
    TIMEOUT_RESPONSE,
    OpenClawConversationAgent,
)


class _Entry:
    """Minimal config entry for agent unit tests."""

    data = {
        CONF_BASE_URL: "http://openclaw.local:18789",
        CONF_API_KEY: "secret",
    }
    options = {
        CONF_MODEL: "openclaw:test",
        CONF_AGENT_ID: DEFAULT_AGENT_ID,
        CONF_TIMEOUT: 30,
        CONF_SYSTEM_PROMPT: "Be useful.",
        CONF_STRIP_EMOJI: True,
        CONF_SESSION_KEY: DEFAULT_SESSION_KEY,
    }


class _Hass:
    """Minimal Home Assistant object for agent unit tests."""


class _Agent(OpenClawConversationAgent):
    """Test agent with a controllable OpenClaw response."""

    def __init__(self, response: str | BaseException) -> None:
        """Initialize the test agent."""
        super().__init__(cast(HomeAssistant, _Hass()), cast(ConfigEntry, _Entry()))
        self.response = response
        self.calls: list[dict[str, object]] = []
        self.chat_log: list[tuple[str, str]] = []

    async def _call_openclaw(
        self,
        text: str,
        principal: dict[str, str],
        language: str,
        conversation_id: str,
    ) -> str:
        """Return a controlled response instead of calling OpenClaw."""
        self.calls.append(
            {
                "text": text,
                "session_key": self._session_key,
                "principal": principal,
                "language": language,
                "conversation_id": conversation_id,
            }
        )
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    def _record_chat_log(
        self,
        user_input: ConversationInput,
        conversation_id: str,
        response_text: str,
    ) -> None:
        """Capture chat log writes instead of using Home Assistant internals."""
        self.chat_log.append((conversation_id, response_text))


class _Response:
    """Async context manager for fake OpenClaw responses."""

    def __init__(self, status: int, body: str) -> None:
        """Initialize the fake response."""
        self.status = status
        self.body = body

    async def __aenter__(self) -> _Response:
        """Enter the fake response context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Exit the fake response context."""

    async def text(self) -> str:
        """Return the fake response body."""
        return self.body


class _Session:
    """Fake aiohttp client session for OpenClaw call tests."""

    def __init__(self, response: _Response) -> None:
        """Initialize the fake session."""
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: aiohttp.ClientTimeout,
    ) -> _Response:
        """Capture an OpenClaw request and return the fake response."""
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self.response


class _ChatSessionContext:
    """Fake chat session context manager."""

    def __enter__(self) -> str:
        """Enter the fake chat session."""
        return "chat-session"

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Exit the fake chat session."""


class _ChatLog:
    """Fake chat log recorder."""

    def __init__(self) -> None:
        """Initialize recorded assistant content."""
        self.contents: list[object] = []

    def async_add_assistant_content_without_tools(self, content: object) -> None:
        """Record assistant content."""
        self.contents.append(content)


class _ChatLogContext:
    """Fake chat log context manager."""

    def __init__(self, chat_log: _ChatLog) -> None:
        """Initialize with a fake chat log."""
        self.chat_log = chat_log

    def __enter__(self) -> _ChatLog:
        """Enter the fake chat log context."""
        return self.chat_log

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Exit the fake chat log context."""


def _conversation_input(
    *,
    text: str = "Turn on the kitchen lights",
    conversation_id: str | None = "conversation-1",
    language: str = "en",
    user_id: str | None = "user-1",
    device_id: str | None = "device-1",
) -> ConversationInput:
    """Build a conversation input for agent tests."""
    return ConversationInput(
        text=text,
        context=Context(user_id=user_id),
        conversation_id=conversation_id,
        device_id=device_id,
        satellite_id=None,
        language=language,
        agent_id="openclaw",
    )


def _speech(result: ConversationResult) -> str:
    """Extract plain speech from a conversation result."""
    data = result.as_dict()
    speech = cast(dict[str, Any], data["response"])["speech"]
    plain = cast(dict[str, Any], speech)["plain"]
    return cast(str, cast(dict[str, Any], plain)["speech"])


@pytest.mark.asyncio
async def test_async_process_returns_response_and_records_chat_log() -> None:
    """Return OpenClaw speech, strip emoji, and record the response."""
    agent = _Agent("Done ✅")
    user_input = _conversation_input()

    result = await agent.async_process(user_input)

    assert result.conversation_id == "conversation-1"
    assert _speech(result) == "Done "
    assert agent.calls == [
        {
            "text": "Turn on the kitchen lights",
            "session_key": DEFAULT_SESSION_KEY,
            "principal": {"user_id": "user-1", "device_id": "device-1"},
            "language": "en",
            "conversation_id": "conversation-1",
        }
    ]
    assert agent.chat_log == [("conversation-1", "Done ")]


@pytest.mark.asyncio
async def test_async_process_uses_session_key_without_conversation_id() -> None:
    """Use the stable OpenClaw session when Home Assistant starts a fresh window."""
    agent = _Agent("Done")

    result = await agent.async_process(_conversation_input(conversation_id=None))

    assert result.conversation_id == DEFAULT_SESSION_KEY
    assert agent.calls[0]["session_key"] == DEFAULT_SESSION_KEY
    assert agent.chat_log == [(DEFAULT_SESSION_KEY, "Done")]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (asyncio.TimeoutError(), TIMEOUT_RESPONSE),
        (aiohttp.ClientError("boom"), NETWORK_ERROR_RESPONSE),
        (RuntimeError("boom"), GENERIC_ERROR_RESPONSE),
    ],
)
@pytest.mark.asyncio
async def test_async_process_returns_fallback_speech(
    error: BaseException,
    expected: str,
) -> None:
    """Return stable fallback speech for expected failure classes."""
    agent = _Agent(error)

    result = await agent.async_process(_conversation_input())

    assert _speech(result) == expected
    assert agent.chat_log == [("conversation-1", expected)]


@pytest.mark.asyncio
async def test_async_process_propagates_cancellation() -> None:
    """Let Home Assistant cancellation propagate."""
    agent = _Agent(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await agent.async_process(_conversation_input())

    assert agent.chat_log == []


def test_build_timeout_with_limit() -> None:
    """Build bounded request timeouts from configured seconds."""
    agent = _Agent("Done")

    timeout = agent._build_timeout()

    assert timeout.total == 30
    assert timeout.connect == 10
    assert timeout.sock_connect == 10
    assert timeout.sock_read is None


def test_build_timeout_without_total_limit() -> None:
    """Use an unbounded total timeout when configured as zero."""
    entry = _Entry()
    entry.options = {**entry.options, CONF_TIMEOUT: 0}
    agent = OpenClawConversationAgent(
        cast(HomeAssistant, _Hass()),
        cast(ConfigEntry, entry),
    )

    timeout = agent._build_timeout()

    assert timeout.total is None
    assert timeout.connect == 10
    assert timeout.sock_connect == 10
    assert timeout.sock_read is None


def test_attribution_and_supported_languages() -> None:
    """Expose attribution and English support metadata."""
    agent = _Agent("Done")

    assert agent.attribution == {
        "name": "Powered by OpenClaw",
        "url": "https://openclaw.ai",
    }
    assert agent.supported_languages == ["en"]


def test_record_chat_log_uses_home_assistant_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record assistant content through Home Assistant chat log helpers."""
    sessions: list[tuple[object, str]] = []
    logs: list[tuple[object, object, object]] = []
    chat_log = _ChatLog()

    def fake_get_chat_session(
        hass: object, conversation_id: str
    ) -> _ChatSessionContext:
        sessions.append((hass, conversation_id))
        return _ChatSessionContext()

    def fake_get_chat_log(
        hass: object,
        session: object,
        user_input: object,
    ) -> _ChatLogContext:
        logs.append((hass, session, user_input))
        return _ChatLogContext(chat_log)

    monkeypatch.setattr(
        conversation_module,
        "async_get_chat_session",
        fake_get_chat_session,
    )
    monkeypatch.setattr(conversation_module, "async_get_chat_log", fake_get_chat_log)
    agent = OpenClawConversationAgent(
        cast(HomeAssistant, _Hass()),
        cast(ConfigEntry, _Entry()),
    )
    user_input = _conversation_input()

    agent._record_chat_log(user_input, "conversation-1", "Hello")

    assert sessions == [(agent.hass, "conversation-1")]
    assert logs == [(agent.hass, "chat-session", user_input)]
    assert len(chat_log.contents) == 1
    content = cast(Any, chat_log.contents[0])
    assert content.agent_id == "openclaw"
    assert content.content == "Hello"


@pytest.mark.asyncio
async def test_call_openclaw_sends_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send OpenClaw chat-completions payloads through HA's shared session."""
    session = _Session(
        _Response(
            200,
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
            'data: {"choices":[{"delta":{"content":" there"}}]}\n',
        )
    )
    monkeypatch.setattr(
        conversation_module,
        "async_get_clientsession",
        lambda hass: session,
    )
    agent = OpenClawConversationAgent(
        cast(HomeAssistant, _Hass()),
        cast(ConfigEntry, _Entry()),
    )

    result = await agent._call_openclaw(
        "Hi",
        {"user_id": "user-1", "device_id": "device-1"},
        "en",
        "conversation-1",
    )

    assert result == "Hello there"
    assert session.calls[0]["url"] == (
        "http://openclaw.local:18789/v1/chat/completions"
    )
    assert session.calls[0]["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "x-openclaw-session-key": f"{DEFAULT_SESSION_KEY}:conversation-1",
        "x-openclaw-message-channel": "homeassistant",
    }
    payload = cast(dict[str, object], session.calls[0]["json"])
    assert payload["model"] == "openclaw:test"
    assert payload["messages"] == [
        {"role": "system", "content": "Be useful."},
        {"role": "user", "content": "Hi"},
    ]
    assert payload["stream"] is True
    assert payload["language"] == "en"
    assert payload["conversation_id"] == f"{DEFAULT_SESSION_KEY}:conversation-1"
    assert payload["user"] == f"{DEFAULT_SESSION_KEY}:conversation-1"
    assert payload["home_assistant_conversation_id"] == "conversation-1"
    assert "agent_id" not in payload
    assert payload["user_id"] == "user-1"
    assert payload["device_id"] == "device-1"
    assert isinstance(payload["local_date"], str)
    assert cast(aiohttp.ClientTimeout, session.calls[0]["timeout"]).total == 30


@pytest.mark.asyncio
async def test_call_openclaw_omits_empty_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not send an empty system message."""
    entry = _Entry()
    entry.options = {**entry.options, CONF_SYSTEM_PROMPT: ""}
    session = _Session(_Response(200, '{"choices":[{"message":{"content":"ok"}}]}'))
    monkeypatch.setattr(
        conversation_module,
        "async_get_clientsession",
        lambda hass: session,
    )
    agent = OpenClawConversationAgent(
        cast(HomeAssistant, _Hass()),
        cast(ConfigEntry, entry),
    )

    assert (
        await agent._call_openclaw(
            "Hi",
            {"user_id": "", "device_id": ""},
            "en",
            DEFAULT_SESSION_KEY,
        )
        == "ok"
    )

    payload = cast(dict[str, object], session.calls[0]["json"])
    assert payload["messages"] == [{"role": "user", "content": "Hi"}]


@pytest.mark.asyncio
async def test_call_openclaw_sends_configured_agent_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin OpenClaw requests to a configured agent when one is set."""
    entry = _Entry()
    entry.options = {**entry.options, CONF_AGENT_ID: "homeops"}
    session = _Session(_Response(200, '{"choices":[{"message":{"content":"ok"}}]}'))
    monkeypatch.setattr(
        conversation_module,
        "async_get_clientsession",
        lambda hass: session,
    )
    agent = OpenClawConversationAgent(
        cast(HomeAssistant, _Hass()),
        cast(ConfigEntry, entry),
    )

    assert (
        await agent._call_openclaw(
            "Hi",
            {"user_id": "", "device_id": ""},
            "en",
            "conversation-1",
        )
        == "ok"
    )

    expected_session_key = "agent:homeops:conversation-1"
    assert session.calls[0]["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "x-openclaw-session-key": expected_session_key,
        "x-openclaw-message-channel": "homeassistant",
    }
    payload = cast(dict[str, object], session.calls[0]["json"])
    assert payload["agent_id"] == "homeops"
    assert payload["conversation_id"] == expected_session_key
    assert payload["user"] == expected_session_key
    assert payload["home_assistant_conversation_id"] == "conversation-1"


@pytest.mark.asyncio
async def test_call_openclaw_raises_for_non_success_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a readable error when OpenClaw returns a non-200 response."""
    session = _Session(_Response(500, "x" * 250))
    monkeypatch.setattr(
        conversation_module,
        "async_get_clientsession",
        lambda hass: session,
    )
    agent = OpenClawConversationAgent(
        cast(HomeAssistant, _Hass()),
        cast(ConfigEntry, _Entry()),
    )

    with pytest.raises(RuntimeError, match=f"OpenClaw returned 500: {'x' * 200}"):
        await agent._call_openclaw(
            "Hi",
            {"user_id": "", "device_id": ""},
            "en",
            DEFAULT_SESSION_KEY,
        )


def test_normalize_session_key() -> None:
    """Normalize empty session keys back to the default."""
    assert OpenClawConversationAgent._normalize_session_key(" custom ") == "custom"
    assert (
        OpenClawConversationAgent._normalize_session_key(DEFAULT_SESSION_KEY, "homeops")
        == "agent:homeops:home-assistant-assist"
    )
    assert (
        OpenClawConversationAgent._normalize_session_key(" custom ", "homeops")
        == "custom"
    )
    assert OpenClawConversationAgent._normalize_session_key("") == DEFAULT_SESSION_KEY
    assert OpenClawConversationAgent._normalize_session_key(None) == DEFAULT_SESSION_KEY


def test_compose_openclaw_session_key() -> None:
    """Combine OpenClaw session config with Home Assistant session ids."""
    assert (
        OpenClawConversationAgent._compose_openclaw_session_key(
            DEFAULT_SESSION_KEY,
            DEFAULT_SESSION_KEY,
        )
        == DEFAULT_SESSION_KEY
    )
    assert (
        OpenClawConversationAgent._compose_openclaw_session_key(
            DEFAULT_SESSION_KEY,
            "conversation-1",
        )
        == f"{DEFAULT_SESSION_KEY}:conversation-1"
    )
    assert (
        OpenClawConversationAgent._compose_openclaw_session_key(
            "agent:homeops:home-assistant-assist",
            "conversation-1",
        )
        == "agent:homeops:conversation-1"
    )
