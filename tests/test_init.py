"""Tests for OpenClaw integration setup helpers."""

from __future__ import annotations

from typing import Any, cast

import pytest
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

import custom_components.openclaw_conversation as integration
from custom_components.openclaw_conversation import (
    _async_update_options,
    async_migrate_entry,
    async_setup,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.openclaw_conversation.const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_STRIP_EMOJI,
    CONF_SYSTEM_PROMPT,
    CONF_TIMEOUT,
)


class _ConfigEntries:
    """Capture config entry update calls."""

    def __init__(self) -> None:
        """Initialize captured update state."""
        self.update: dict[str, Any] | None = None
        self.reloaded: list[str] = []

    def async_update_entry(self, entry: _Entry, **kwargs: Any) -> bool:
        """Record an update_entry call."""
        self.update = kwargs
        return True

    async def async_reload(self, entry_id: str) -> None:
        """Record reload calls."""
        self.reloaded.append(entry_id)


class _Hass:
    """Minimal Home Assistant object for migration tests."""

    def __init__(self) -> None:
        """Initialize the fake config entries manager."""
        self.config_entries = _ConfigEntries()


class _Entry:
    """Minimal config entry object for migration tests."""

    version = 1
    entry_id = "entry-1"
    unique_id = None
    data = {
        CONF_BASE_URL: "http://localhost:18789",
        CONF_API_KEY: "secret",
        CONF_MODEL: "openclaw:legacy",
        CONF_TIMEOUT: 90,
        CONF_SYSTEM_PROMPT: "Legacy prompt",
    }
    options = {CONF_STRIP_EMOJI: False}

    def __init__(self) -> None:
        """Initialize unload callback tracking."""
        self.update_listener: Any = None
        self.unload_callbacks: list[Any] = []

    def add_update_listener(self, listener: Any) -> str:
        """Capture the registered update listener."""
        self.update_listener = listener
        return "remove-listener"

    def async_on_unload(self, callback: Any) -> None:
        """Capture unload callbacks."""
        self.unload_callbacks.append(callback)


class _NewerEntry(_Entry):
    """Config entry from a future integration version."""

    version = 3


class _CurrentEntry(_Entry):
    """Current config entry that should not be migrated."""

    version = 2


@pytest.mark.asyncio
async def test_migrate_v1_entry_moves_behavior_settings_to_options() -> None:
    """Move legacy behavior settings out of config entry data."""
    hass = _Hass()
    entry = _Entry()

    assert await async_migrate_entry(
        cast(HomeAssistant, hass),
        cast(ConfigEntry, entry),
    )

    assert hass.config_entries.update == {
        "data": {
            CONF_BASE_URL: "http://localhost:18789",
            CONF_API_KEY: "secret",
        },
        "options": {
            CONF_MODEL: "openclaw:legacy",
            CONF_TIMEOUT: 90,
            CONF_SYSTEM_PROMPT: "Legacy prompt",
            CONF_STRIP_EMOJI: False,
        },
        "unique_id": "http://localhost:18789",
        "version": 2,
    }


@pytest.mark.asyncio
async def test_migrate_current_entry_noops() -> None:
    """Leave current config entries unchanged."""
    hass = _Hass()

    assert await async_migrate_entry(
        cast(HomeAssistant, hass),
        cast(ConfigEntry, _CurrentEntry()),
    )
    assert hass.config_entries.update is None


@pytest.mark.asyncio
async def test_migrate_future_entry_fails() -> None:
    """Reject config entries from newer integration versions."""
    hass = _Hass()

    assert not await async_migrate_entry(
        cast(HomeAssistant, hass),
        cast(ConfigEntry, _NewerEntry()),
    )
    assert hass.config_entries.update is None


@pytest.mark.asyncio
async def test_async_setup_returns_true() -> None:
    """Set up the integration from YAML/config entry only schema."""
    assert await async_setup(cast(HomeAssistant, _Hass()), {}) is True


@pytest.mark.asyncio
async def test_setup_entry_registers_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register a conversation agent and update listener."""
    calls: list[tuple[str, object, object]] = []

    def fake_set_agent(
        hass: HomeAssistant,
        entry: ConfigEntry,
        agent: object,
    ) -> None:
        calls.append(("set", entry, agent))

    monkeypatch.setattr(integration, "async_set_agent", fake_set_agent)
    hass = _Hass()
    entry = _Entry()

    assert await async_setup_entry(
        cast(HomeAssistant, hass),
        cast(ConfigEntry, entry),
    )

    assert calls[0][0] == "set"
    assert calls[0][1] is entry
    assert isinstance(calls[0][2], integration.OpenClawConversationAgent)
    assert entry.update_listener is not None
    assert entry.unload_callbacks == ["remove-listener"]


@pytest.mark.asyncio
async def test_update_options_reloads_entry() -> None:
    """Reload the entry when options change."""
    hass = _Hass()
    entry = _Entry()

    await _async_update_options(cast(HomeAssistant, hass), cast(ConfigEntry, entry))

    assert hass.config_entries.reloaded == ["entry-1"]


@pytest.mark.asyncio
async def test_unload_entry_unregisters_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the conversation agent during unload."""
    calls: list[tuple[object, object]] = []

    def fake_unset_agent(hass: HomeAssistant, entry: ConfigEntry) -> None:
        calls.append((hass, entry))

    monkeypatch.setattr(integration, "async_unset_agent", fake_unset_agent)
    hass = _Hass()
    entry = _Entry()

    assert await async_unload_entry(
        cast(HomeAssistant, hass),
        cast(ConfigEntry, entry),
    )
    assert calls == [(hass, entry)]
