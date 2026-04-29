"""Tests for OpenClaw config flow helper behavior."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import aiohttp
import pytest
import voluptuous as vol
from aiohttp.client_reqrep import ConnectionKey
from homeassistant.config_entries import ConfigEntry, ConfigFlowResult
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import UNDEFINED, UndefinedType

import custom_components.openclaw_conversation.config_flow as config_flow_module
from custom_components.openclaw_conversation.config_flow import (
    OpenClawConversationConfigFlow,
    OpenClawOptionsFlowHandler,
    _build_data_schema,
    _connection_data_from_user_input,
    _error_from_validation_response,
    _looks_like_model_error,
    _options_from_user_input,
    _validation_timeout_from_options,
)
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
    DEFAULT_MODEL,
    DEFAULT_SESSION_KEY,
    DEFAULT_STRIP_EMOJI,
    DEFAULT_SYSTEM_PROMPT,
)


class _Entry:
    """Minimal config entry for options flow tests."""

    data = {
        CONF_BASE_URL: "http://localhost:18789",
        CONF_API_KEY: "secret",
        CONF_MODEL: "data-model",
    }
    options = {
        CONF_MODEL: "option-model",
        CONF_AGENT_ID: DEFAULT_AGENT_ID,
        CONF_TIMEOUT: 12,
        CONF_SYSTEM_PROMPT: "Option prompt",
        CONF_STRIP_EMOJI: False,
        CONF_SESSION_KEY: DEFAULT_SESSION_KEY,
    }


class _ExistingEntry(_Entry):
    """Minimal existing entry for reconfigure tests."""

    entry_id = "entry-1"
    title = "OpenClaw"
    unique_id = "http://localhost:18789"


class _OtherEntry(_Entry):
    """Minimal different entry for duplicate reconfigure tests."""

    entry_id = "entry-2"


class _ConfigFlow(OpenClawConversationConfigFlow):
    """Config flow with HA plumbing replaced by deterministic test hooks."""

    def __init__(
        self,
        *,
        errors: dict[str, str] | None = None,
        existing_entry: ConfigEntry | None = None,
        reconfigure_entry: ConfigEntry | None = None,
    ) -> None:
        """Initialize the test flow."""
        super().__init__()
        self.validation_errors = errors or {}
        self.existing_entry = existing_entry
        self.reconfigure_entry = reconfigure_entry or cast(
            ConfigEntry, _ExistingEntry()
        )
        self.unique_ids: list[str | None] = []

    async def async_set_unique_id(
        self,
        unique_id: str | None = None,
        *,
        raise_on_progress: bool = True,
    ) -> ConfigEntry | None:
        """Capture unique IDs instead of consulting Home Assistant."""
        self.unique_ids.append(unique_id)
        return self.existing_entry

    def _abort_if_unique_id_configured(
        self,
        updates: dict[str, Any] | None = None,
        reload_on_update: bool = True,
        *,
        error: str = "already_configured",
        description_placeholders: Mapping[str, str] | None = None,
    ) -> None:
        """Avoid Home Assistant entry registry lookups in tests."""

    def _get_reconfigure_entry(self) -> ConfigEntry:
        """Return the configured reconfigure entry."""
        return self.reconfigure_entry

    async def _async_validate_connection(
        self,
        data: dict[str, str],
        options: dict[str, Any],
    ) -> dict[str, str]:
        """Return controlled validation errors."""
        return self.validation_errors

    def async_update_reload_and_abort(
        self,
        entry: ConfigEntry,
        *,
        unique_id: str | None | UndefinedType = UNDEFINED,
        title: str | UndefinedType = UNDEFINED,
        data: Mapping[str, Any] | UndefinedType = UNDEFINED,
        data_updates: Mapping[str, Any] | UndefinedType = UNDEFINED,
        options: Mapping[str, Any] | UndefinedType = UNDEFINED,
        reason: str | UndefinedType = UNDEFINED,
        reload_even_if_entry_is_unchanged: bool = True,
    ) -> ConfigFlowResult:
        """Capture reconfigure updates instead of updating Home Assistant."""
        return cast(
            ConfigFlowResult,
            {
                "type": "abort",
                "reason": "reconfigure_successful" if reason is UNDEFINED else reason,
                "unique_id": unique_id,
                "data_updates": data_updates,
                "options": options,
            },
        )


class _Response:
    """Async context manager for fake validation responses."""

    def __init__(
        self,
        status: int,
        *,
        body: str = "",
        text_error: BaseException | None = None,
    ) -> None:
        """Initialize a fake aiohttp response."""
        self.status = status
        self.reason = "OK"
        self.body = body
        self.text_error = text_error

    async def __aenter__(self) -> _Response:
        """Enter the response context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Exit the response context."""

    async def text(self) -> str:
        """Return the fake response body."""
        if self.text_error is not None:
            raise self.text_error
        return self.body


class _Session:
    """Fake aiohttp client session for validation tests."""

    def __init__(self, response: _Response | BaseException) -> None:
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
        """Record a validation request and return the fake response."""
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def _connector_error() -> aiohttp.ClientConnectorError:
    """Build a realistic aiohttp connector error."""
    connection_key = ConnectionKey(
        host="localhost",
        port=18789,
        is_ssl=False,
        ssl=False,
        proxy=None,
        proxy_auth=None,
        proxy_headers_hash=None,
    )
    return aiohttp.ClientConnectorError(connection_key, OSError("boom"))


def test_detects_model_error() -> None:
    """Detect common gateway messages for an unavailable model."""
    assert _looks_like_model_error("Model openclaw:test not found")
    assert _looks_like_model_error("unknown model")
    assert not _looks_like_model_error("authentication failed")


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (200, "", None),
        (400, "bad payload", "bad_request"),
        (400, "model does not exist", "model_not_available"),
        (401, "", "invalid_auth"),
        (403, "", "forbidden"),
        (404, "", "endpoint_not_found"),
        (405, "", "endpoint_disabled"),
        (500, "boom", "server_error"),
        (503, "unknown model", "model_not_available"),
        (418, "", "cannot_connect"),
    ],
)
def test_error_from_validation_response(
    status: int,
    body: str,
    expected: str | None,
) -> None:
    """Map gateway validation statuses to config flow errors."""
    assert _error_from_validation_response(status, body) == expected


def test_connection_data_trims_base_url() -> None:
    """Store only connection fields in config entry data."""
    data = _connection_data_from_user_input(
        {
            CONF_NAME: "OpenClaw",
            CONF_BASE_URL: "http://localhost:18789/",
            CONF_API_KEY: "secret",
            CONF_MODEL: "openclaw:main",
        }
    )

    assert data == {
        CONF_BASE_URL: "http://localhost:18789",
        CONF_API_KEY: "secret",
    }


def test_options_from_user_input() -> None:
    """Store behavior settings in config entry options."""
    options = _options_from_user_input(
        {
            CONF_MODEL: "openclaw:other",
            CONF_AGENT_ID: "homeops",
            CONF_TIMEOUT: 45,
            CONF_SYSTEM_PROMPT: "Be concise.",
            CONF_SESSION_KEY: "home-assistant-custom",
        }
    )

    assert options == {
        CONF_MODEL: "openclaw:other",
        CONF_AGENT_ID: "homeops",
        CONF_TIMEOUT: 45,
        CONF_SYSTEM_PROMPT: "Be concise.",
        CONF_STRIP_EMOJI: DEFAULT_STRIP_EMOJI,
        CONF_SESSION_KEY: "home-assistant-custom",
    }


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, 120),
        ({CONF_TIMEOUT: 0}, 120),
        ({CONF_TIMEOUT: 10}, 30),
        ({CONF_TIMEOUT: 44}, 44),
        ({CONF_TIMEOUT: "bad"}, 120),
    ],
)
def test_validation_timeout_from_options(
    options: dict[str, object],
    expected: int,
) -> None:
    """Derive a bounded setup validation timeout from entry options."""
    assert _validation_timeout_from_options(options) == expected


def test_build_data_schema_applies_setup_defaults() -> None:
    """Apply setup form defaults while keeping submitted connection fields."""
    schema = _build_data_schema()

    data = schema({CONF_BASE_URL: "http://localhost:18789/", CONF_API_KEY: "secret"})

    assert data == {
        CONF_NAME: "OpenClaw",
        CONF_BASE_URL: "http://localhost:18789/",
        CONF_API_KEY: "secret",
        CONF_MODEL: DEFAULT_MODEL,
        CONF_AGENT_ID: DEFAULT_AGENT_ID,
        CONF_TIMEOUT: 0,
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_SESSION_KEY: DEFAULT_SESSION_KEY,
    }


def test_build_data_schema_uses_reconfigure_defaults() -> None:
    """Apply existing entry values as defaults for reconfigure."""
    schema = _build_data_schema(
        include_name=False,
        base_url="http://old-host:18789",
        api_key="old-secret",
        model="openclaw:old",
        agent_id="homeops",
        timeout=30,
        system_prompt="Old prompt",
        session_key="home-assistant-old",
    )

    data = schema({})

    assert data == {
        CONF_BASE_URL: "http://old-host:18789",
        CONF_API_KEY: "old-secret",
        CONF_MODEL: "openclaw:old",
        CONF_AGENT_ID: "homeops",
        CONF_TIMEOUT: 30,
        CONF_SYSTEM_PROMPT: "Old prompt",
        CONF_SESSION_KEY: "home-assistant-old",
    }


def test_build_data_schema_rejects_negative_timeout() -> None:
    """Reject timeout values below zero."""
    schema = _build_data_schema()

    with pytest.raises(vol.MultipleInvalid, match="too small"):
        schema(
            {
                CONF_BASE_URL: "http://localhost:18789",
                CONF_API_KEY: "secret",
                CONF_TIMEOUT: -1,
            }
        )


@pytest.mark.asyncio
async def test_options_flow_shows_existing_values() -> None:
    """Show options form defaults merged from config entry data and options."""
    flow = OpenClawOptionsFlowHandler(cast(ConfigEntry, _Entry()))

    result = cast(dict[str, Any], await flow.async_step_init())

    assert result["type"].value == "form"
    assert result["step_id"] == "init"
    schema = cast(vol.Schema, result["data_schema"])
    assert schema({}) == {
        CONF_MODEL: "option-model",
        CONF_AGENT_ID: DEFAULT_AGENT_ID,
        CONF_TIMEOUT: 12,
        CONF_SYSTEM_PROMPT: "Option prompt",
        CONF_SESSION_KEY: DEFAULT_SESSION_KEY,
        CONF_STRIP_EMOJI: False,
    }


def test_config_flow_returns_options_flow() -> None:
    """Return the options flow handler for a config entry."""
    handler = OpenClawConversationConfigFlow.async_get_options_flow(
        cast(ConfigEntry, _Entry())
    )

    assert isinstance(handler, OpenClawOptionsFlowHandler)


@pytest.mark.asyncio
async def test_options_flow_creates_entry_with_defaults() -> None:
    """Persist submitted options and fill omitted defaults."""
    flow = OpenClawOptionsFlowHandler(cast(ConfigEntry, _Entry()))

    result = cast(
        dict[str, Any],
        await flow.async_step_init({CONF_MODEL: "openclaw:new"}),
    )

    assert result["type"].value == "create_entry"
    assert result["title"] == ""
    assert result["data"] == {
        CONF_MODEL: "openclaw:new",
        CONF_AGENT_ID: DEFAULT_AGENT_ID,
        CONF_TIMEOUT: 0,
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STRIP_EMOJI: True,
        CONF_SESSION_KEY: DEFAULT_SESSION_KEY,
    }


@pytest.mark.asyncio
async def test_user_step_shows_form() -> None:
    """Show the initial setup form when no input is submitted."""
    flow = _ConfigFlow()

    result = cast(dict[str, Any], await flow.async_step_user())

    assert result["type"].value == "form"
    assert result["step_id"] == "user"
    assert isinstance(result["data_schema"], vol.Schema)


@pytest.mark.asyncio
async def test_user_step_creates_entry() -> None:
    """Create an entry with connection data and behavior options."""
    flow = _ConfigFlow()

    result = cast(
        dict[str, Any],
        await flow.async_step_user(
            {
                CONF_NAME: "My OpenClaw",
                CONF_BASE_URL: "http://localhost:18789/",
                CONF_API_KEY: "secret",
                CONF_MODEL: "openclaw:new",
                CONF_AGENT_ID: "homeops",
                CONF_TIMEOUT: 22,
                CONF_SYSTEM_PROMPT: "Be brief.",
            }
        ),
    )

    assert result["type"].value == "create_entry"
    assert result["title"] == "My OpenClaw"
    assert result["data"] == {
        CONF_BASE_URL: "http://localhost:18789",
        CONF_API_KEY: "secret",
    }
    assert result["options"] == {
        CONF_MODEL: "openclaw:new",
        CONF_AGENT_ID: "homeops",
        CONF_TIMEOUT: 22,
        CONF_SYSTEM_PROMPT: "Be brief.",
        CONF_STRIP_EMOJI: True,
        CONF_SESSION_KEY: DEFAULT_SESSION_KEY,
    }
    assert flow.unique_ids == ["http://localhost:18789"]


@pytest.mark.asyncio
async def test_user_step_returns_validation_errors() -> None:
    """Redisplay setup form when gateway validation fails."""
    flow = _ConfigFlow(errors={"base": "cannot_connect"})

    result = cast(
        dict[str, Any],
        await flow.async_step_user(
            {
                CONF_BASE_URL: "http://localhost:18789",
                CONF_API_KEY: "secret",
            }
        ),
    )

    assert result["type"].value == "form"
    assert result["errors"] == {"base": "cannot_connect"}


@pytest.mark.asyncio
async def test_reconfigure_step_shows_existing_values() -> None:
    """Show reconfigure form using merged entry data and options."""
    flow = _ConfigFlow()

    result = cast(dict[str, Any], await flow.async_step_reconfigure())

    assert result["type"].value == "form"
    assert result["step_id"] == "reconfigure"
    schema = cast(vol.Schema, result["data_schema"])
    assert schema({}) == {
        CONF_BASE_URL: "http://localhost:18789",
        CONF_API_KEY: "secret",
        CONF_MODEL: "option-model",
        CONF_AGENT_ID: DEFAULT_AGENT_ID,
        CONF_TIMEOUT: 12,
        CONF_SYSTEM_PROMPT: "Option prompt",
        CONF_SESSION_KEY: DEFAULT_SESSION_KEY,
    }


@pytest.mark.asyncio
async def test_reconfigure_step_aborts_duplicate_url() -> None:
    """Abort reconfigure when the submitted URL belongs to another entry."""
    flow = _ConfigFlow(existing_entry=cast(ConfigEntry, _OtherEntry()))

    result = cast(
        dict[str, Any],
        await flow.async_step_reconfigure(
            {
                CONF_BASE_URL: "http://other-host:18789",
                CONF_API_KEY: "secret",
            }
        ),
    )

    assert result["type"].value == "abort"
    assert result["reason"] == "already_configured"


@pytest.mark.asyncio
async def test_reconfigure_step_updates_entry() -> None:
    """Return reconfigure updates after successful validation."""
    flow = _ConfigFlow()

    result = cast(
        dict[str, Any],
        await flow.async_step_reconfigure(
            {
                CONF_BASE_URL: "http://new-host:18789/",
                CONF_API_KEY: "new-secret",
                CONF_MODEL: "openclaw:new",
                CONF_AGENT_ID: "homeops",
                CONF_TIMEOUT: 44,
                CONF_SYSTEM_PROMPT: "New prompt",
            }
        ),
    )

    assert result == {
        "type": "abort",
        "reason": "reconfigure_successful",
        "unique_id": "http://new-host:18789",
        "data_updates": {
            CONF_BASE_URL: "http://new-host:18789",
            CONF_API_KEY: "new-secret",
        },
        "options": {
            CONF_MODEL: "openclaw:new",
            CONF_AGENT_ID: "homeops",
            CONF_TIMEOUT: 44,
            CONF_SYSTEM_PROMPT: "New prompt",
            CONF_STRIP_EMOJI: True,
            CONF_SESSION_KEY: DEFAULT_SESSION_KEY,
        },
    }


@pytest.mark.asyncio
async def test_validate_connection_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return no errors when the gateway accepts the validation request."""
    flow = OpenClawConversationConfigFlow()
    flow.hass = cast(HomeAssistant, object())
    session = _Session(_Response(200))
    monkeypatch.setattr(
        config_flow_module,
        "async_get_clientsession",
        lambda hass: session,
    )

    result = await flow._async_validate_connection(
        {CONF_BASE_URL: "http://localhost:18789", CONF_API_KEY: "secret"},
        {CONF_MODEL: "openclaw:test"},
    )

    assert result == {}
    assert session.calls[0]["url"] == "http://localhost:18789/v1/chat/completions"
    assert session.calls[0]["json"] == {
        "model": "openclaw:test",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "max_tokens": 1,
    }
    assert session.calls[0]["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
    }
    assert cast(aiohttp.ClientTimeout, session.calls[0]["timeout"]).total == 120


@pytest.mark.asyncio
async def test_validate_connection_sends_agent_id_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Include a configured OpenClaw agent id in setup validation."""
    flow = OpenClawConversationConfigFlow()
    flow.hass = cast(HomeAssistant, object())
    session = _Session(_Response(200))
    monkeypatch.setattr(
        config_flow_module,
        "async_get_clientsession",
        lambda hass: session,
    )

    result = await flow._async_validate_connection(
        {CONF_BASE_URL: "http://localhost:18789", CONF_API_KEY: "secret"},
        {CONF_MODEL: "openclaw:test", CONF_AGENT_ID: "homeops"},
    )

    assert result == {}
    assert cast(dict[str, object], session.calls[0]["json"])["agent_id"] == "homeops"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (_Response(400, body="bad payload"), {"base": "bad_request"}),
        (_Response(400, body="model not found"), {"base": "model_not_available"}),
        (_Response(500, body="boom"), {"base": "server_error"}),
        (_Response(418, body="short and stout"), {"base": "cannot_connect"}),
        (
            _Response(400, text_error=aiohttp.ClientError("body failed")),
            {"base": "bad_request"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_validate_connection_status_errors(
    monkeypatch: pytest.MonkeyPatch,
    response: _Response,
    expected: dict[str, str],
) -> None:
    """Map gateway response statuses to validation errors."""
    flow = OpenClawConversationConfigFlow()
    flow.hass = cast(HomeAssistant, object())
    monkeypatch.setattr(
        config_flow_module,
        "async_get_clientsession",
        lambda hass: _Session(response),
    )

    result = await flow._async_validate_connection(
        {CONF_BASE_URL: "http://localhost:18789", CONF_API_KEY: "secret"},
        {},
    )

    assert result == expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_connector_error(), {"base": "cannot_reach"}),
        (TimeoutError(), {"base": "timeout"}),
        (aiohttp.ClientError("boom"), {"base": "cannot_connect"}),
    ],
)
@pytest.mark.asyncio
async def test_validate_connection_client_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    expected: dict[str, str],
) -> None:
    """Map aiohttp validation exceptions to config flow errors."""
    flow = OpenClawConversationConfigFlow()
    flow.hass = cast(HomeAssistant, object())
    monkeypatch.setattr(
        config_flow_module,
        "async_get_clientsession",
        lambda hass: _Session(error),
    )

    result = await flow._async_validate_connection(
        {CONF_BASE_URL: "http://localhost:18789", CONF_API_KEY: "secret"},
        {},
    )

    assert result == expected
