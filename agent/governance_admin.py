"""Budget-independent governance control plane for gateway and local operators."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from agent.cost_context_governance import _InterProcessLock
from hermes_cli.config import atomic_config_write, read_raw_config
from hermes_constants import get_hermes_home


_MODES = {"block", "log"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _platform_name(source: Any) -> str:
    value = getattr(source, "platform", "")
    return str(getattr(value, "value", value) or "").strip().lower()


def _subject_hash(subject: str) -> str:
    return hashlib.sha256(subject.encode("utf-8")).hexdigest()


def _governance_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    value = raw_config.get("cost_context_governance")
    return value if isinstance(value, dict) else {}


def _effective_policy(raw_mode: Any) -> str:
    mode = str(raw_mode or "disabled").strip().lower()
    if mode in {"block", "enforce"}:
        return "block"
    if mode in {"log", "observe"}:
        return "log"
    return "disabled"


def _is_explicit_admin(config: dict[str, Any], *, platform: str, subject: str) -> bool:
    administrators = config.get("administrators")
    if not isinstance(administrators, dict) or not platform or not subject:
        return False
    allowed = administrators.get(platform)
    if not isinstance(allowed, (list, tuple, set)):
        return False
    return subject in {str(item).strip() for item in allowed if str(item).strip()}


def _append_audit(home: Path, payload: dict[str, Any]) -> None:
    path = home / "governance" / "admin-audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def execute_gateway_command(
    text: str,
    *,
    source: Any,
    controllers: Iterable[Any] = (),
) -> str | None:
    """Handle an exact ``!governance`` command without invoking an agent or model."""
    raw = str(text or "").strip()
    parts = raw.split()
    if not parts or parts[0].lower() != "!governance":
        return None
    platform = _platform_name(source)
    subject = str(getattr(source, "user_id", "") or "").strip()
    home = get_hermes_home()
    config_path = home / "config.yaml"
    raw_config = read_raw_config()
    governance = _governance_config(raw_config)
    authorized = _is_explicit_admin(governance, platform=platform, subject=subject)
    audit = {
        "timestamp": _now(),
        "event": "governance_admin_command",
        "platform": platform,
        "subject_sha256": _subject_hash(subject),
        "profile": home.name if home.parent.name == "profiles" else "default",
        "command": parts[1:] if len(parts) > 1 else [],
        "authorized": authorized,
    }
    if not authorized:
        audit["outcome"] = "denied"
        _append_audit(home, audit)
        return "Comando de governança rejeitado: identidade administrativa não autorizada para esta plataforma/perfil."

    if parts[1:] == ["status"]:
        policy = _effective_policy(governance.get("mode"))
        audit.update({"outcome": "status", "effective_policy": policy})
        _append_audit(home, audit)
        return f"Governança: política efetiva `{policy}`; perfil `{audit['profile']}`."

    if len(parts) == 3 and parts[1] == "mode" and parts[2] in _MODES:
        requested = parts[2]
        lock_path = config_path.with_suffix(config_path.suffix + ".governance.lock")
        with _InterProcessLock(lock_path):
            fresh_config = read_raw_config()
            fresh_governance = _governance_config(fresh_config)
            if not _is_explicit_admin(fresh_governance, platform=platform, subject=subject):
                audit["outcome"] = "denied_after_recheck"
                _append_audit(home, audit)
                return "Comando de governança rejeitado: autorização administrativa mudou durante a operação."
            previous = _effective_policy(fresh_governance.get("mode"))
            fresh_governance["mode"] = requested
            fresh_config["cost_context_governance"] = fresh_governance
            atomic_config_write(config_path, fresh_config, sort_keys=False)
        updated = 0
        for controller in controllers:
            if controller is None or not hasattr(controller, "set_mode"):
                continue
            controller.set_mode(requested)
            updated += 1
        audit.update({
            "outcome": "updated",
            "previous_policy": previous,
            "effective_policy": requested,
            "active_controllers_updated": updated,
        })
        _append_audit(home, audit)
        return (
            f"Governança atualizada: política efetiva `{requested}`; perfil `{audit['profile']}`; "
            f"persistida em config.yaml."
        )

    audit["outcome"] = "invalid"
    _append_audit(home, audit)
    return "Formato: `!governance status` ou `!governance mode log|block`."


def _local_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local out-of-band governance administration")
    parser.add_argument("action", choices=("status", "mode"))
    parser.add_argument("mode", nargs="?", choices=tuple(sorted(_MODES)))
    args = parser.parse_args(argv)
    home = get_hermes_home()
    config_path = home / "config.yaml"
    raw_config = read_raw_config()
    governance = _governance_config(raw_config)
    if args.action == "status":
        print(json.dumps({"profile_home": str(home), "effective_policy": _effective_policy(governance.get("mode"))}))
        return 0
    if not args.mode:
        parser.error("mode requires log or block")
    lock_path = config_path.with_suffix(config_path.suffix + ".governance.lock")
    with _InterProcessLock(lock_path):
        raw_config = read_raw_config()
        governance = _governance_config(raw_config)
        previous = _effective_policy(governance.get("mode"))
        governance["mode"] = args.mode
        raw_config["cost_context_governance"] = governance
        atomic_config_write(config_path, raw_config, sort_keys=False)
    _append_audit(home, {
        "timestamp": _now(), "event": "governance_admin_local", "authorized": True,
        "previous_policy": previous, "effective_policy": args.mode,
        "subject_sha256": _subject_hash(str(os.getuid()) if hasattr(os, "getuid") else getpass.getuser()),
    })
    print(json.dumps({"profile_home": str(home), "effective_policy": args.mode, "persisted": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_local_command())
