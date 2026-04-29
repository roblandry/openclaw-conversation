# AGENTS.md

Guidance for coding agents working in this repository.

## Project

This is a Home Assistant custom integration named `openclaw_conversation`.
It registers OpenClaw as a Home Assistant conversation agent and forwards user
messages to an OpenAI-compatible OpenClaw Gateway chat completions endpoint.

The integration is distributed for HACS.

## Commands

Assume a local `.venv` unless noted. Use `.venv/bin/<tool>` when the venv is not
activated.

- Install dev dependencies: `.venv/bin/pip install -r requirements.txt`
- Run all tests: `.venv/bin/pytest -q`
- Run one test: `.venv/bin/pytest tests/test_config_flow.py::test_name -q`
- Lint: `.venv/bin/ruff check .`
- Format: `.venv/bin/ruff format .`
- Type check: `.venv/bin/pyright`
- Pre-commit suite: `.venv/bin/pre-commit run --all-files`

## Repository Layout

- `custom_components/openclaw_conversation/`: Home Assistant integration package.
- `conversation.py`: conversation agent, OpenClaw request payloads, response
  parsing, timeout handling, and chat log updates.
- `config_flow.py`: UI setup/options flow and gateway validation.
- `const.py`: domain, config keys, and defaults.
- `translations/`: setup/options UI text for the custom integration.
- `tests/`: pytest coverage once added.

## Architecture Notes

The integration uses Home Assistant's conversation agent manager via
`conversation.async_set_agent`. It does not create entities or devices.

Use Home Assistant's shared aiohttp client session with
`async_get_clientsession(hass)` for outbound requests. Do not create standalone
`aiohttp.ClientSession` objects inside the integration.

Keep the Home Assistant side focused on request shaping and response parsing.
OpenClaw should own routing, memory, session policy, and tool execution.

## Testing Expectations

Add focused pytest coverage for:

- Config flow success and gateway error mapping.
- Options flow defaults and persistence.
- Conversation request payload construction.
- Streaming and non-streaming OpenAI-style response parsing.
- Timeout/network/error speech behavior.

Before handoff on meaningful changes, run `ruff check .`, `ruff format --check .`,
and the smallest relevant pytest target. Run the full pytest suite once tests
exist.

`pytest-homeassistant-custom-component` is installed for future HA fixture tests,
but the normal suite disables its auto-loaded `homeassistant` pytest plugin with
`-p no:homeassistant`. With Python 3.14.2/HA 2026.4.4, the plugin currently hangs
during teardown while waiting on a `pycares` shutdown thread. Do not remove that
pytest option until the plugin-backed tests have an explicit cleanup strategy.

## Agent Guardrails

- Do not revert user changes in this working tree.
- Do not commit local caches, virtualenvs, `.codex`, or `__pycache__`.
- Do not log API tokens or full sensitive request payloads.
- Preserve backwards compatibility with existing config entries when moving
  values between entry data and options.
