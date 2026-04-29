"""Config flow for OpenClaw Conversation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import aiohttp
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntry, ConfigFlowResult
from homeassistant.const import CONF_NAME
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    NumberSelector,  # pyright: ignore[reportUnknownVariableType]
    NumberSelectorConfig,
    NumberSelectorMode,
    TextSelector,  # pyright: ignore[reportUnknownVariableType]
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CONF_AGENT_ID,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_SESSION_KEY,
    CONF_STRIP_EMOJI,
    CONF_SYSTEM_PROMPT,
    CONF_TIMEOUT,
    DEFAULT_AGENT_ID,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SESSION_KEY,
    DEFAULT_STRIP_EMOJI,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TIMEOUT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

_DEFAULT_NAME = "OpenClaw"
_MIN_VALIDATION_TIMEOUT = 30
_DEFAULT_VALIDATION_TIMEOUT = 120
_SelectorValidator = Callable[[object], object]


def _text_selector(config: TextSelectorConfig) -> _SelectorValidator:
    """Return a typed wrapper for Home Assistant text selectors."""
    return cast(_SelectorValidator, TextSelector(config))


def _number_selector(config: NumberSelectorConfig) -> _SelectorValidator:
    """Return a typed wrapper for Home Assistant number selectors."""
    return cast(_SelectorValidator, NumberSelector(config))


def _looks_like_model_error(body: str) -> bool:
    """Detect gateway errors caused by an unknown or unavailable model."""
    if not body:
        return False
    lowered = body.lower()
    if "model" not in lowered:
        return False
    return any(
        hint in lowered
        for hint in (
            "not found",
            "not available",
            "unknown",
            "invalid",
            "does not exist",
            "no such",
        )
    )


def _error_from_validation_response(status: int, body: str) -> str | None:
    """Map an OpenClaw validation response to a config flow error reason."""
    if status == 200:
        return None
    if status == 400:
        return "model_not_available" if _looks_like_model_error(body) else "bad_request"
    if status == 401:
        return "invalid_auth"
    if status == 403:
        return "forbidden"
    if status == 404:
        return "endpoint_not_found"
    if status == 405:
        return "endpoint_disabled"
    if 500 <= status < 600:
        return (
            "model_not_available" if _looks_like_model_error(body) else "server_error"
        )
    return "cannot_connect"


def _connection_data_from_user_input(user_input: dict[str, Any]) -> dict[str, str]:
    """Return config entry data required to connect to OpenClaw."""
    return {
        CONF_BASE_URL: str(user_input[CONF_BASE_URL]).rstrip("/"),
        CONF_API_KEY: str(user_input[CONF_API_KEY]),
    }


def _options_from_user_input(user_input: dict[str, Any]) -> dict[str, Any]:
    """Return configurable OpenClaw behavior options."""
    return {
        CONF_MODEL: user_input.get(CONF_MODEL, DEFAULT_MODEL),
        CONF_AGENT_ID: str(user_input.get(CONF_AGENT_ID, DEFAULT_AGENT_ID)).strip(),
        CONF_TIMEOUT: user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
        CONF_SYSTEM_PROMPT: user_input.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
        CONF_STRIP_EMOJI: user_input.get(CONF_STRIP_EMOJI, DEFAULT_STRIP_EMOJI),
        CONF_SESSION_KEY: str(
            user_input.get(CONF_SESSION_KEY, DEFAULT_SESSION_KEY)
        ).strip(),
    }


def _validation_timeout_from_options(options: dict[str, Any]) -> int:
    """Return the setup validation timeout in seconds."""
    try:
        configured_timeout = int(options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        configured_timeout = DEFAULT_TIMEOUT

    if configured_timeout <= 0:
        return _DEFAULT_VALIDATION_TIMEOUT
    return max(configured_timeout, _MIN_VALIDATION_TIMEOUT)


def _build_data_schema(
    *,
    name: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    agent_id: str = DEFAULT_AGENT_ID,
    timeout: int = DEFAULT_TIMEOUT,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    session_key: str = DEFAULT_SESSION_KEY,
    include_name: bool = True,
) -> vol.Schema:
    """Build the shared config flow form schema."""
    schema: dict[Any, Any] = {}
    if include_name:
        schema[vol.Optional(CONF_NAME, default=name or _DEFAULT_NAME)] = str
    api_key_marker = (
        vol.Required(CONF_API_KEY, default=api_key)
        if api_key is not None
        else vol.Required(CONF_API_KEY)
    )

    schema.update(
        {
            vol.Required(
                CONF_BASE_URL,
                default=base_url,
            ): _text_selector(TextSelectorConfig(type=TextSelectorType.URL)),
            api_key_marker: _text_selector(
                TextSelectorConfig(type=TextSelectorType.PASSWORD)
            ),
            vol.Optional(CONF_MODEL, default=model): str,
            vol.Optional(
                CONF_AGENT_ID,
                default=agent_id,
            ): _text_selector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            vol.Optional(CONF_TIMEOUT, default=timeout): vol.All(
                _number_selector(
                    NumberSelectorConfig(
                        min=0,
                        step=1,
                        mode=NumberSelectorMode.BOX,
                    )
                ),
                vol.Coerce(int),
            ),
            vol.Optional(
                CONF_SYSTEM_PROMPT,
                default=system_prompt,
            ): _text_selector(TextSelectorConfig(multiline=True)),
            vol.Optional(
                CONF_SESSION_KEY,
                default=session_key,
            ): _text_selector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        }
    )
    return vol.Schema(schema)


class OpenClawConversationConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenClaw Conversation."""

    VERSION = 2

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OpenClawOptionsFlowHandler:
        """Get the options flow handler."""
        return OpenClawOptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            data = _connection_data_from_user_input(user_input)
            options = _options_from_user_input(user_input)
            await self.async_set_unique_id(data[CONF_BASE_URL])
            self._abort_if_unique_id_configured()

            errors = await self._async_validate_connection(data, options)

            if not errors:
                name = user_input.get(CONF_NAME, _DEFAULT_NAME)
                return self.async_create_entry(
                    title=name,
                    data=data,
                    options=options,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=_build_data_schema(),
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle changes to connection settings for an existing config entry."""
        entry = self._get_reconfigure_entry()
        config = {**entry.data, **entry.options}
        errors: dict[str, str] = {}

        if user_input is not None:
            data = _connection_data_from_user_input(user_input)
            options = _options_from_user_input(user_input)
            existing_entry = await self.async_set_unique_id(data[CONF_BASE_URL])
            if existing_entry is not None and existing_entry.entry_id != entry.entry_id:
                return self.async_abort(reason="already_configured")

            errors = await self._async_validate_connection(data, options)
            if not errors:
                return self.async_update_reload_and_abort(
                    entry,
                    unique_id=data[CONF_BASE_URL],
                    data_updates=data,
                    options=options,
                )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=_build_data_schema(
                name=entry.title,
                base_url=config.get(CONF_BASE_URL, DEFAULT_BASE_URL),
                api_key=config.get(CONF_API_KEY),
                model=config.get(CONF_MODEL, DEFAULT_MODEL),
                agent_id=config.get(CONF_AGENT_ID, DEFAULT_AGENT_ID),
                timeout=config.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                system_prompt=config.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
                session_key=config.get(CONF_SESSION_KEY, DEFAULT_SESSION_KEY),
                include_name=False,
            ),
            errors=errors,
        )

    async def _async_validate_connection(
        self,
        data: dict[str, str],
        options: dict[str, Any],
    ) -> dict[str, str]:
        """Validate the OpenClaw gateway connection."""
        base_url = data[CONF_BASE_URL]
        url = f"{base_url}/v1/chat/completions"
        try:
            session = async_get_clientsession(self.hass)
            headers = {
                "Authorization": f"Bearer {data[CONF_API_KEY]}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": options.get(CONF_MODEL, DEFAULT_MODEL),
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "max_tokens": 1,
            }
            if agent_id := options.get(CONF_AGENT_ID):
                payload["agent_id"] = agent_id
            validation_timeout = _validation_timeout_from_options(options)
            _LOGGER.debug(
                "Validating OpenClaw Gateway: POST %s (model=%s, timeout=%ss)",
                url,
                payload["model"],
                validation_timeout,
            )
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=validation_timeout),
            ) as resp:
                body = ""
                if resp.status != 200:
                    try:
                        body = await resp.text()
                    except aiohttp.ClientError:
                        body = ""
                _LOGGER.debug(
                    "OpenClaw Gateway validation response: %s %s - body=%s",
                    resp.status,
                    resp.reason,
                    body[:500],
                )
                if (
                    error := _error_from_validation_response(resp.status, body)
                ) is None:
                    return {}
                if error == "cannot_connect":
                    _LOGGER.warning(
                        "OpenClaw Gateway returned unexpected status %s for %s - body=%s",
                        resp.status,
                        url,
                        body[:500],
                    )
                return {"base": error}
        except aiohttp.ClientConnectorError as err:
            _LOGGER.warning("OpenClaw Gateway unreachable at %s: %s", url, err)
            return {"base": "cannot_reach"}
        except TimeoutError:
            _LOGGER.warning("OpenClaw Gateway validation timed out at %s", url)
            return {"base": "timeout"}
        except aiohttp.ClientError as err:
            _LOGGER.warning("OpenClaw Gateway client error at %s: %s", url, err)

        return {"base": "cannot_connect"}


class OpenClawOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options for OpenClaw Conversation."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(
                title="",
                data=_options_from_user_input(user_input),
            )

        config = {**self._config_entry.data, **self._config_entry.options}
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MODEL,
                        default=config.get(CONF_MODEL, DEFAULT_MODEL),
                    ): str,
                    vol.Optional(
                        CONF_AGENT_ID,
                        default=config.get(CONF_AGENT_ID, DEFAULT_AGENT_ID),
                    ): _text_selector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                    vol.Optional(
                        CONF_TIMEOUT,
                        default=config.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                    ): vol.All(
                        _number_selector(
                            NumberSelectorConfig(
                                min=0,
                                step=1,
                                mode=NumberSelectorMode.BOX,
                            )
                        ),
                        vol.Coerce(int),
                    ),
                    vol.Optional(
                        CONF_SYSTEM_PROMPT,
                        default=config.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
                    ): _text_selector(TextSelectorConfig(multiline=True)),
                    vol.Optional(
                        CONF_SESSION_KEY,
                        default=config.get(CONF_SESSION_KEY, DEFAULT_SESSION_KEY),
                    ): _text_selector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                    vol.Optional(
                        CONF_STRIP_EMOJI,
                        default=config.get(CONF_STRIP_EMOJI, DEFAULT_STRIP_EMOJI),
                    ): bool,
                }
            ),
        )
