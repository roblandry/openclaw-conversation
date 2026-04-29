"""Conversation agent for OpenClaw."""

from __future__ import annotations

import asyncio
import json as json_mod
import logging
import re
import time
from collections.abc import Mapping
from typing import Any, TypeAlias, cast

import aiohttp
from homeassistant.components.conversation import (
    AssistantContent,
    async_get_chat_log,  # pyright: ignore[reportUnknownVariableType]
)
from homeassistant.components.conversation.models import (
    AbstractConversationAgent,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.util import dt as dt_util, ulid

from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_SESSION_KEY,
    CONF_STRIP_EMOJI,
    CONF_SYSTEM_PROMPT,
    CONF_TIMEOUT,
    DEFAULT_MODEL,
    DEFAULT_SESSION_KEY,
    DEFAULT_STRIP_EMOJI,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TIMEOUT,
)

_LOGGER = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["en"]
TIMEOUT_RESPONSE = "OpenClaw took too long to respond."
NETWORK_ERROR_RESPONSE = "Network error while contacting OpenClaw."
GENERIC_ERROR_RESPONSE = "Error communicating with OpenClaw."

_EMOJI_PATTERN = re.compile(
    "[\u2600-\u27bf\ufe0f\U0001f000-\U0001faff]",
    flags=re.UNICODE,
)
_STREAM_PREFIX = "data: "

JsonObject: TypeAlias = Mapping[str, object]
ChatMessage: TypeAlias = dict[str, str]


def _as_json_object(value: object) -> JsonObject | None:
    """Return value as a string-keyed mapping when it is JSON-object shaped."""
    if isinstance(value, Mapping):
        return cast(JsonObject, value)
    return None


def _extract_error_message(error: object) -> str:
    """Return a readable error message from an OpenAI-compatible error object."""
    if (error_obj := _as_json_object(error)) is not None:
        message = error_obj.get("message") or error_obj.get("code")
        if isinstance(message, str):
            return message
        return json_mod.dumps(error)
    return str(error)


def _extract_stream_content(raw: str) -> tuple[str, str | None, bool]:
    """Extract assistant text from an OpenAI-compatible server-sent stream."""
    content = ""
    stream_error: str | None = None
    saw_done = False

    for line in raw.splitlines():
        line = line.strip()
        if not line or not line.startswith(_STREAM_PREFIX):
            continue
        data_str = line[len(_STREAM_PREFIX) :]
        if data_str == "[DONE]":
            saw_done = True
            break
        try:
            chunk: object = json_mod.loads(data_str)
        except json_mod.JSONDecodeError:
            continue
        chunk_obj = _as_json_object(chunk)
        if chunk_obj is None:
            continue
        if "error" in chunk_obj:
            stream_error = _extract_error_message(chunk_obj["error"])
            continue
        choices = chunk_obj.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choices_list = cast(list[object], choices)
        choice = choices_list[0]
        choice_obj = _as_json_object(choice)
        if choice_obj is None:
            continue
        delta = choice_obj.get("delta")
        delta_obj = _as_json_object(delta)
        if delta_obj is None:
            continue
        delta_content = delta_obj.get("content")
        if isinstance(delta_content, str):
            content += delta_content

    return content, stream_error, saw_done


def _extract_json_content(raw: str) -> tuple[str, str | None]:
    """Extract assistant text from a non-streaming OpenAI-compatible response."""
    try:
        data: object = json_mod.loads(raw)
    except json_mod.JSONDecodeError:
        return "", None

    data_obj = _as_json_object(data)
    if data_obj is None:
        return "", None

    if "error" in data_obj:
        return "", _extract_error_message(data_obj["error"])

    choices = data_obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", None
    choices_list = cast(list[object], choices)
    choice = choices_list[0]
    choice_obj = _as_json_object(choice)
    if choice_obj is None:
        return "", None
    message = choice_obj.get("message")
    message_obj = _as_json_object(message)
    if message_obj is None:
        return "", None
    content = message_obj.get("content")
    return content if isinstance(content, str) else "", None


def _parse_openclaw_response(raw: str) -> str:
    """Return assistant content from an OpenClaw chat-completions response."""
    content, stream_error, saw_done = _extract_stream_content(raw)
    if not content and raw:
        content, json_error = _extract_json_content(raw)
        stream_error = stream_error or json_error

    if content:
        return content

    if stream_error:
        raise RuntimeError(f"OpenClaw returned an error: {stream_error}")
    if saw_done:
        raise RuntimeError(
            "OpenClaw returned an empty stream (received [DONE] with no content). "
            "This usually means the gateway timed out before the agent produced a "
            "response. Increase agents.defaults.llm.idleTimeoutSeconds in "
            "openclaw.json (e.g. 180) and check the gateway logs."
        )
    raise RuntimeError(f"No response from OpenClaw. Raw: {raw[:500]}")


class OpenClawConversationAgent(AbstractConversationAgent):
    """OpenClaw conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        config = {**entry.data, **entry.options}
        self._base_url = config[CONF_BASE_URL]
        self._api_key = config[CONF_API_KEY]
        self._model = config.get(CONF_MODEL, DEFAULT_MODEL)
        self._timeout = self._normalize_timeout(
            config.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
        )
        self._system_prompt = config.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)
        self._strip_emoji = config.get(CONF_STRIP_EMOJI, DEFAULT_STRIP_EMOJI)
        self._session_key = self._normalize_session_key(
            config.get(CONF_SESSION_KEY, DEFAULT_SESSION_KEY)
        )

    @property
    def attribution(self) -> dict[str, str]:
        """Return attribution."""
        return {"name": "Powered by OpenClaw", "url": "https://openclaw.ai"}

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return SUPPORTED_LANGUAGES

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a sentence."""
        conversation_id = user_input.conversation_id or ulid.ulid_now()
        principal = self._resolve_principal(user_input)

        try:
            start = time.monotonic()
            response_text = await self._call_openclaw(
                user_input.text,
                principal,
                user_input.language,
            )
            elapsed = time.monotonic() - start
            _LOGGER.info(
                "OpenClaw responded in %.1fs (%d chars)",
                elapsed,
                len(response_text),
            )
        except asyncio.TimeoutError:
            _LOGGER.error("OpenClaw request timed out after %ds", self._timeout)
            response_text = TIMEOUT_RESPONSE
        except aiohttp.ClientError as err:
            _LOGGER.error("Network error calling OpenClaw: %s", err)
            response_text = NETWORK_ERROR_RESPONSE
        except asyncio.CancelledError:
            _LOGGER.warning("OpenClaw request was cancelled by Home Assistant")
            raise
        except Exception as err:
            _LOGGER.error("Error calling OpenClaw: %s: %s", type(err).__name__, err)
            response_text = GENERIC_ERROR_RESPONSE

        if self._strip_emoji:
            response_text = _EMOJI_PATTERN.sub("", response_text)

        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(response_text)

        self._record_chat_log(user_input, conversation_id, response_text)

        return ConversationResult(
            response=response,
            conversation_id=conversation_id,
        )

    def _record_chat_log(
        self,
        user_input: ConversationInput,
        conversation_id: str,
        response_text: str,
    ) -> None:
        """Record the assistant response in Home Assistant's conversation log."""
        with async_get_chat_session(self.hass, conversation_id) as session:
            with async_get_chat_log(self.hass, session, user_input) as chat_log:
                chat_log.async_add_assistant_content_without_tools(
                    AssistantContent(
                        agent_id=user_input.agent_id,
                        content=response_text,
                    )
                )

    def _resolve_principal(self, user_input: ConversationInput) -> dict[str, str]:
        """Extract stable HA identity fields when available."""
        context = getattr(user_input, "context", None)
        user_id = getattr(context, "user_id", None) if context else None
        device_id = getattr(user_input, "device_id", None)
        return {
            "user_id": user_id or "",
            "device_id": device_id or "",
        }

    @staticmethod
    def _normalize_session_key(value: object) -> str:
        """Normalize the configured OpenClaw session key."""
        if isinstance(value, str) and value.strip():
            return value.strip()
        return DEFAULT_SESSION_KEY

    @staticmethod
    def _normalize_timeout(value: Any) -> int:
        """Normalize the configured timeout to a non-negative integer."""
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return DEFAULT_TIMEOUT

    def _build_timeout(self) -> aiohttp.ClientTimeout:
        """Build the request timeout configuration."""
        connect_timeout = 10
        if self._timeout <= 0:
            return aiohttp.ClientTimeout(
                total=None,
                connect=connect_timeout,
                sock_connect=connect_timeout,
                sock_read=None,
            )

        total_timeout = max(int(self._timeout), 1)
        return aiohttp.ClientTimeout(
            total=total_timeout,
            connect=min(connect_timeout, total_timeout),
            sock_connect=min(connect_timeout, total_timeout),
            sock_read=None,
        )

    async def _call_openclaw(
        self,
        text: str,
        principal: dict[str, str],
        language: str,
    ) -> str:
        """Call OpenClaw chat completions API with streaming."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "x-openclaw-session-key": self._session_key,
            "x-openclaw-message-channel": "homeassistant",
        }

        api_messages: list[ChatMessage] = []
        if self._system_prompt:
            api_messages.append({"role": "system", "content": self._system_prompt})
        api_messages.append({"role": "user", "content": text})

        payload: dict[str, object] = {
            "model": self._model,
            "messages": api_messages,
            "stream": True,
            "language": language,
            "local_date": dt_util.now().date().isoformat(),
            "conversation_id": self._session_key,
            "user": self._session_key,
            "user_id": principal["user_id"],
            "device_id": principal["device_id"],
        }
        timeout = self._build_timeout()
        session = async_get_clientsession(self.hass)
        async with session.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"OpenClaw returned {resp.status}: {body[:200]}")

            raw = await resp.text()
            _LOGGER.debug("OpenClaw raw response: %s", raw)
            return _parse_openclaw_response(raw)
