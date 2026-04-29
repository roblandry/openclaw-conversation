"""Repository metadata tests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "custom_components" / "openclaw_conversation"


def test_manifest_matches_custom_integration() -> None:
    """Validate basic Home Assistant custom integration metadata."""
    manifest = json.loads((INTEGRATION / "manifest.json").read_text())

    assert manifest["domain"] == "openclaw_conversation"
    assert manifest["name"] == "OpenClaw Conversation"
    assert manifest["config_flow"] is True
    assert manifest["integration_type"] == "service"
    assert manifest["iot_class"] == "local_polling"
    assert manifest["requirements"] == []
    assert manifest["documentation"].startswith(
        "https://github.com/roblandry/openclaw-conversation"
    )


def test_translations_include_config_and_options() -> None:
    """Ensure bundled translations expose the expected config flow sections."""
    translations = json.loads((INTEGRATION / "translations" / "en.json").read_text())

    assert not (INTEGRATION / "strings.json").exists()
    assert "user" in translations["config"]["step"]
    assert "reconfigure" in translations["config"]["step"]
    assert "init" in translations["options"]["step"]
    assert "model_not_available" in translations["config"]["error"]
