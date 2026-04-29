"""OpenClaw Conversation integration for Home Assistant."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Final, cast

from homeassistant.components.conversation import async_set_agent, async_unset_agent
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_STRIP_EMOJI,
    CONF_SYSTEM_PROMPT,
    CONF_TIMEOUT,
    DEFAULT_MODEL,
    DEFAULT_STRIP_EMOJI,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TIMEOUT,
    DOMAIN,
)
from .conversation import OpenClawConversationAgent

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA: Final[Callable[[ConfigType], ConfigType]] = cast(
    Callable[[ConfigType], ConfigType],
    cv.config_entry_only_config_schema(DOMAIN),  # pyright: ignore[reportUnknownMemberType]
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OpenClaw Conversation."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OpenClaw Conversation from a config entry."""
    agent = OpenClawConversationAgent(hass, entry)

    async_set_agent(hass, entry, agent)
    entry.async_on_unload(entry.add_update_listener(_async_update_options))
    _LOGGER.info("OpenClaw Conversation agent registered")
    return True


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate older OpenClaw Conversation config entries."""
    if entry.version > 2:
        return False

    if entry.version == 1:
        new_data = dict(entry.data)
        new_options: dict[str, Any] = {
            CONF_MODEL: new_data.pop(CONF_MODEL, DEFAULT_MODEL),
            CONF_TIMEOUT: new_data.pop(CONF_TIMEOUT, DEFAULT_TIMEOUT),
            CONF_SYSTEM_PROMPT: new_data.pop(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
            CONF_STRIP_EMOJI: entry.options.get(CONF_STRIP_EMOJI, DEFAULT_STRIP_EMOJI),
            **entry.options,
        }
        hass.config_entries.async_update_entry(
            entry,
            data=new_data,
            options=new_options,
            unique_id=entry.unique_id or new_data.get(CONF_BASE_URL),
            version=2,
        )
        _LOGGER.debug("Migrated OpenClaw Conversation config entry to version 2")

    return True


async def _async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update — reload the integration."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenClaw Conversation."""
    async_unset_agent(hass, entry)
    return True
