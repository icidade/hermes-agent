import json
from types import SimpleNamespace

from agent import governance_admin


def _source(user_id="owner-1", platform="telegram"):
    return SimpleNamespace(user_id=user_id, platform=platform)


def test_unauthorized_governance_command_is_rejected_without_write(tmp_path, monkeypatch):
    monkeypatch.setattr(governance_admin, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        governance_admin,
        "read_raw_config",
        lambda: {"cost_context_governance": {"mode": "block", "administrators": {"telegram": ["owner-1"]}}},
    )
    writes = []
    monkeypatch.setattr(governance_admin, "atomic_config_write", lambda *args, **kwargs: writes.append(args))

    response = governance_admin.execute_gateway_command(
        "!governance mode log", source=_source("intruder")
    )

    assert "rejeitado" in response
    assert writes == []
    audit = json.loads((tmp_path / "governance" / "admin-audit.jsonl").read_text().splitlines()[-1])
    assert audit["authorized"] is False
    assert "intruder" not in json.dumps(audit)


def test_authorized_mode_change_persists_and_updates_live_controller(tmp_path, monkeypatch):
    raw = {"cost_context_governance": {"mode": "block", "administrators": {"slack": ["U123"]}}}
    monkeypatch.setattr(governance_admin, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(governance_admin, "read_raw_config", lambda: raw)
    writes = []
    monkeypatch.setattr(
        governance_admin,
        "atomic_config_write",
        lambda path, data, **kwargs: writes.append((path, data.copy(), kwargs)),
    )
    controller = SimpleNamespace(mode="block")
    controller.set_mode = lambda mode: setattr(controller, "mode", mode)

    response = governance_admin.execute_gateway_command(
        "!governance mode log", source=_source("U123", "slack"), controllers=[controller]
    )

    assert "política efetiva `log`" in response
    assert raw["cost_context_governance"]["mode"] == "log"
    assert controller.mode == "log"
    assert len(writes) == 1


def test_governance_status_uses_legacy_mode_alias_without_write(tmp_path, monkeypatch):
    monkeypatch.setattr(governance_admin, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        governance_admin,
        "read_raw_config",
        lambda: {"cost_context_governance": {"mode": "observe", "administrators": {"telegram": ["owner-1"]}}},
    )
    response = governance_admin.execute_gateway_command(
        "!governance status", source=_source()
    )
    assert "política efetiva `log`" in response
