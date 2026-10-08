"""Contracts for noop-key in-place config refresh (F1, #1625).

A PUT that changes only keys a channel declares in ``hot_reload_noop_keys``
(or the manager bool overrides) is applied in place through
``ChannelManager.refresh_channel_config`` — the adapter is never stopped — and
the route reports ``applied: "refreshed"``. Any other key, the ``enabled``
transition, or a rejected refresh falls back to the full hot-swap path.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api_server
import src.channels.manager as manager_module
from src.api import channels_config_routes as routes
from src.channels.base import BaseChannel
from src.channels.bus.queue import MessageBus
from src.channels.feishu import FeishuChannel, FeishuConfig
from src.channels.manager import ChannelManager

CLIENT_ID = "ding-client-id-1234567890"
STORED_SECRET = "stored-secret-abcdefghij"


# --------------------------------------------------------------------------- #
# BaseChannel.refresh_config (planbook tests 1 + 2)
# --------------------------------------------------------------------------- #


def test_refresh_config_swaps_values_on_real_adapter() -> None:
    """A valid section re-validates and swaps in place, keeping the model type."""
    section_a = FeishuChannel.default_config()
    channel = FeishuChannel(section_a, MessageBus())
    assert channel.config.react_emoji == "THUMBSUP"

    section_b = dict(section_a, react_emoji="DONE", done_emoji="OK", tool_hint_prefix="*")
    assert channel.refresh_config(section_b) is True

    assert channel.config.react_emoji == "DONE"
    assert channel.config.done_emoji == "OK"
    assert channel.config.tool_hint_prefix == "*"
    assert type(channel.config) is FeishuConfig


def test_refresh_config_rejects_invalid_section_preserving_old() -> None:
    """A section that fails validation returns False and leaves the old config."""
    section = FeishuChannel.default_config()
    channel = FeishuChannel(section, MessageBus())
    assert channel.refresh_config(dict(section, react_emoji="DONE")) is True
    before = channel.config

    # A list where a str field is declared fails model_validate.
    assert channel.refresh_config(dict(section, react_emoji=[])) is False

    assert channel.config is before
    assert channel.config.react_emoji == "DONE"


# --------------------------------------------------------------------------- #
# ChannelManager.refresh_channel_config (planbook test 3)
# --------------------------------------------------------------------------- #


class _FakeChannel(BaseChannel):
    """Concrete adapter stand-in; inherits the real ``refresh_config``."""

    name = "fake"
    display_name = "Fake"

    def __init__(self, config: Any, bus: MessageBus, **kwargs: Any) -> None:
        del kwargs
        super().__init__(config, bus)
        self.stop_calls = 0

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self._running = False

    async def send(self, msg: Any) -> None:
        del msg


def _fake_inspect_channels(config: Any) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    if not isinstance(config, dict):
        return statuses
    for name, section in config.items():
        if not isinstance(section, dict):
            continue
        statuses[name] = {
            "name": name,
            "available": True,
            "display_name": name.title(),
            "install_hint": "",
            "error": "",
            "configured": True,
            "enabled": bool(section.get("enabled")),
            "loaded": False,
            "running": False,
        }
    return statuses


def _build_manager(monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]) -> ChannelManager:
    monkeypatch.setattr(manager_module, "discover_channel_names", lambda: ["fake"])
    monkeypatch.setattr(manager_module, "load_channel_class", lambda name: _FakeChannel)
    monkeypatch.setattr(manager_module, "inspect_channels", _fake_inspect_channels)
    return ChannelManager(config, MessageBus())


def test_manager_refresh_channel_config_reresolves_bool_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refresh re-resolves the manager bool overrides without stopping the adapter."""

    async def scenario() -> None:
        manager = _build_manager(
            monkeypatch, {"fake": {"enabled": True, "send_progress": True}}
        )
        channel = manager.channels["fake"]
        assert channel.send_progress is True

        result = await manager.refresh_channel_config(
            "fake", {"enabled": True, "send_progress": False}
        )

        assert result is True
        assert channel.send_progress is False
        assert channel.stop_calls == 0
        assert manager.config["fake"] == {"enabled": True, "send_progress": False}

        # An unknown / not-built channel cannot be refreshed.
        empty = _build_manager(monkeypatch, {})
        assert await empty.refresh_channel_config("fake", {"enabled": True}) is False

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# PUT /channels/config/{name} routing (planbook test 4)
# --------------------------------------------------------------------------- #


class _FakeManager:
    """Records refresh/reload calls; ``get_channel`` gates the refresh path."""

    def __init__(self, *, refresh_ok: bool = True, live: bool = True) -> None:
        self.refresh_calls: list[tuple[str, dict]] = []
        self.reload_calls: list[tuple[str, dict | None]] = []
        self._refresh_ok = refresh_ok
        self._live = live

    def get_channel(self, name: str) -> Any:
        return object() if self._live else None

    async def refresh_channel_config(self, name: str, section: dict) -> bool:
        self.refresh_calls.append((name, section))
        return self._refresh_ok

    async def reload_channel(self, name: str, section: dict | None) -> dict[str, Any]:
        self.reload_calls.append((name, section))
        return {"loaded": bool(section and section.get("enabled"))}


class _FakeRuntime:
    """An authoritative running flag + manager, matching ChannelRuntime's surface."""

    def __init__(self, *, running: bool, manager: _FakeManager) -> None:
        self._running = running
        self.manager = manager

    def status(self) -> dict[str, Any]:
        return {"running": self._running, "channels": {}}

    async def start(self, *, start_manager: bool = True) -> None:
        del start_manager

    async def stop(self) -> None:
        self._running = False


def _write_agent_config(tmp_path: Path, channels: dict[str, Any]) -> Path:
    path = tmp_path / "agent.json"
    path.write_text(json.dumps({"channels": channels}), encoding="utf-8")
    return path


def _client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    channels: dict[str, Any],
    runtime: Any,
) -> TestClient:
    _write_agent_config(tmp_path, channels)
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path))
    monkeypatch.setattr(api_server, "_channel_runtime", runtime)
    monkeypatch.setattr(api_server, "_channel_bus", None)
    monkeypatch.setattr(api_server, "_channel_manager", None)
    monkeypatch.setattr(api_server, "_get_session_service", lambda: object())
    return TestClient(api_server.app, client=("127.0.0.1", 50000))


@pytest.fixture(autouse=True)
def _fresh_locks():
    """Give every test unbound locks (asyncio.Lock binds to a loop on contention)."""
    routes._channel_locks.clear()
    routes._runtime_lock = asyncio.Lock()
    yield
    routes._channel_locks.clear()
    routes._runtime_lock = asyncio.Lock()


def _dingtalk_section(**overrides: Any) -> dict[str, Any]:
    section: dict[str, Any] = {
        "enabled": True,
        "client_id": CLIENT_ID,
        "client_secret": STORED_SECRET,
        "allow_from": ["*"],
    }
    section.update(overrides)
    return section


def test_route_noop_only_edit_applies_refreshed_without_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A noop-key-only PUT refreshes in place; the stop+start path never runs."""
    manager = _FakeManager(refresh_ok=True, live=True)
    runtime = _FakeRuntime(running=True, manager=manager)
    client = _client(
        tmp_path,
        monkeypatch,
        channels={"dingtalk": _dingtalk_section()},
        runtime=runtime,
    )

    response = client.put(
        "/channels/config/dingtalk", json={"config": {"allow_from": ["alice"]}}
    )

    assert response.status_code == 200
    assert response.json()["applied"] == "refreshed"
    assert [name for name, _ in manager.refresh_calls] == ["dingtalk"]
    assert manager.reload_calls == []


@pytest.mark.parametrize(
    "put_config, refresh_ok, expect_refresh_called",
    [
        ({"client_secret": "rotated-secret"}, True, False),  # connection key
        ({"enabled": False}, True, False),  # start/stop transition
        ({"allow_from": ["bob"]}, False, True),  # refresh rejected -> fail-safe
    ],
)
def test_route_falls_back_to_hot_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    put_config: dict[str, Any],
    refresh_ok: bool,
    expect_refresh_called: bool,
) -> None:
    """A non-noop key, the enabled transition, or a rejected refresh hot-swaps."""
    manager = _FakeManager(refresh_ok=refresh_ok, live=True)
    runtime = _FakeRuntime(running=True, manager=manager)
    client = _client(
        tmp_path,
        monkeypatch,
        channels={"dingtalk": _dingtalk_section()},
        runtime=runtime,
    )

    response = client.put("/channels/config/dingtalk", json={"config": put_config})

    assert response.status_code == 200
    assert response.json()["applied"] == "hot_swapped"
    assert [name for name, _ in manager.reload_calls] == ["dingtalk"]
    assert bool(manager.refresh_calls) is expect_refresh_called
