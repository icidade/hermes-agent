from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


_LOCK_ACQUIRE_TIMEOUT_S = 30.0
_LOCK_POLL_INTERVAL_S = 0.05
_USAGE_MARGIN_INPUT = 1.15
_USAGE_MARGIN_OUTPUT = 1.20
_BRIEF_PRESERVE_KEYS = (
    "scope",
    "scope_exclusions",
    "decisions",
    "evidence_ids",
    "claim_ids",
    "qa_obligations",
    "artifact_references",
    "open_questions",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return default
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _normalize_optional_identifier(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def estimate_messages_tokens(messages: Iterable[dict]) -> int:
    total = 0
    for msg in messages or []:
        total += estimate_text_tokens(_json_dumps(msg)) + 6
    return total


def hash_context_block(block: Any) -> str:
    return hashlib.sha256(_json_dumps(block).encode("utf-8")).hexdigest()


def _text_metadata(value: Any) -> dict[str, Any]:
    text = "" if value is None else str(value)
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "chars": len(text)}


def _exception_metadata(exc: BaseException | str | None) -> dict[str, Any]:
    text = "" if exc is None else str(exc)
    return {
        "category": type(exc).__name__ if isinstance(exc, BaseException) else "error",
        "fingerprint": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


_SAFE_ARTIFACT_KEYS = {
    "event", "tool_name", "provider", "model", "request_class", "operation",
    "engagement_id", "task_id", "request_id", "reservation_id", "authorization_id", "logical_call_id", "attempt_id", "profile", "profile_key", "agent_role", "governance_role",
    "status", "label", "mode", "phase", "transition", "decision", "state", "state_before",
    "state_after", "termination_reason", "resume_identifier", "resume_token", "category",
    "fingerprint", "path", "checkpoint", "evidence_ledger", "claim_ledger", "artifact_references",
    "context_manifest", "qa_obligations", "evidence_collected", "claims_evaluated", "open_questions",
}


def project_artifact_value(value: Any, *, key: str = "") -> Any:
    """Project persistence payloads to metadata; never retain arbitrary text."""
    lowered = key.lower()
    if lowered in {"messages_tail", "messages", "prompt", "content", "objective", "summary", "task",
                   "result_preview", "error", "error_message", "exception", "recommended_next_step",
                   "limitations", "errors", "tool_args", "tool_result", "authorization_header", "url"}:
        if lowered == "messages_tail" and isinstance(value, list):
            return {"count": len(value), "roles": [str(item.get("role", "")) for item in value if isinstance(item, dict)],
                    "sha256": hash_context_block(value)}
        if lowered == "tool_args" and isinstance(value, dict):
            return {"keys": sorted(str(name) for name in value), "sha256": hash_context_block(value),
                    "chars": len(_json_dumps(value))}
        if isinstance(value, list):
            return {"count": len(value), "sha256": hash_context_block(value)}
        return _text_metadata(value)
    if isinstance(value, dict):
        return {str(name): project_artifact_value(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [project_artifact_value(item, key=key) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if lowered in _SAFE_ARTIFACT_KEYS:
        return str(value)
    return _text_metadata(value)


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {
                key: getattr(value, key)
                for key in vars(value)
                if not key.startswith("_")
            }
        except Exception:
            return {}
    return {}


def _lookup_nested(data: Any, *paths: str) -> Any:
    for path in paths:
        cur = data
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict):
                if part not in cur:
                    ok = False
                    break
                cur = cur.get(part)
            else:
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
        if ok and cur is not None:
            return cur
    return None


@dataclass
class ThresholdConfig:
    informational_ratio: float = 0.50
    warning_ratio: float = 0.70
    approval_ratio: float = 0.85
    hard_stop_ratio: float = 1.00


@dataclass
class ProgressConfig:
    equivalent_tool_calls: int = 3
    same_error_repeats: int = 3
    no_progress_iterations: int = 4
    timeout_checkpoint_seconds: int = 30
    logical_call_max_preflights: int = 8
    logical_call_max_compactions: int = 4
    logical_call_min_progress_tokens: int = 64
    logical_call_min_progress_ratio: float = 0.02


@dataclass
class BudgetProfile:
    total_model_calls: int = 8
    calls_per_agent: int = 6
    calls_per_delegated_task: int = 5
    input_tokens: int = 24000
    output_tokens: int = 12000
    total_tokens: int = 36000
    context_tokens_per_request: int = 12000
    wall_clock_seconds: int = 300
    tool_invocations: int = 25
    retries: int = 4
    delegation_count: int = 3
    iterations_without_progress: int = 4
    qa_reserve_ratio: float = 0.20

    @classmethod
    def from_mapping(cls, data: Optional[dict], fallback: Optional["BudgetProfile"] = None) -> "BudgetProfile":
        base = asdict(fallback or cls())
        for key, value in (data or {}).items():
            if key in base:
                base[key] = value
        return cls(**base)


@dataclass
class GovernanceConfig:
    mode: str = "disabled"
    workspace_dir: str = "engagements"
    default_profile: str = "conversational"
    force_for_profiles: list[str] = field(default_factory=lambda: ["chief-of-staff"])
    engagement_keywords: list[str] = field(default_factory=lambda: [
        "miniciso", "security", "assessment", "review", "threat model", "appsec", "bug bounty",
    ])
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    role_toolsets: dict[str, list[str]] = field(default_factory=dict)
    profiles: dict[str, BudgetProfile] = field(default_factory=lambda: {
        "conversational": BudgetProfile(
            total_model_calls=3,
            calls_per_agent=3,
            calls_per_delegated_task=0,
            input_tokens=6000,
            output_tokens=3000,
            total_tokens=9000,
            context_tokens_per_request=4500,
            wall_clock_seconds=90,
            tool_invocations=4,
            retries=1,
            delegation_count=0,
            iterations_without_progress=2,
            qa_reserve_ratio=0.0,
        ),
        "bounded": BudgetProfile(
            total_model_calls=5,
            calls_per_agent=4,
            calls_per_delegated_task=2,
            input_tokens=12000,
            output_tokens=6000,
            total_tokens=18000,
            context_tokens_per_request=9000,
            wall_clock_seconds=180,
            tool_invocations=10,
            retries=2,
            delegation_count=1,
            iterations_without_progress=3,
            qa_reserve_ratio=0.10,
        ),
        "standard_engagement": BudgetProfile(),
        "deep_engagement": BudgetProfile(
            total_model_calls=16,
            calls_per_agent=10,
            calls_per_delegated_task=8,
            input_tokens=48000,
            output_tokens=24000,
            total_tokens=72000,
            context_tokens_per_request=18000,
            wall_clock_seconds=600,
            tool_invocations=40,
            retries=5,
            delegation_count=6,
            iterations_without_progress=5,
            qa_reserve_ratio=0.25,
        ),
        "custom": BudgetProfile(),
    })

    @classmethod
    def from_mapping(cls, data: Optional[dict]) -> "GovernanceConfig":
        base = cls()
        if not data:
            return base
        threshold = ThresholdConfig(**{**asdict(base.threshold), **(data.get("threshold") or {})})
        progress = ProgressConfig(**{**asdict(base.progress), **(data.get("progress") or {})})
        profiles = dict(base.profiles)
        for key, profile_data in (data.get("profiles") or {}).items():
            profiles[key] = BudgetProfile.from_mapping(profile_data, fallback=profiles.get(key, BudgetProfile()))
        return cls(
            mode=str(data.get("mode", base.mode)),
            workspace_dir=str(data.get("workspace_dir", base.workspace_dir)),
            default_profile=str(data.get("default_profile", base.default_profile)),
            force_for_profiles=list(data.get("force_for_profiles", base.force_for_profiles)),
            engagement_keywords=list(data.get("engagement_keywords", base.engagement_keywords)),
            threshold=threshold,
            progress=progress,
            role_toolsets={str(k): list(v or []) for k, v in dict(data.get("role_toolsets") or {}).items()},
            profiles=profiles,
        )


@dataclass
class EnvelopeUsage:
    model_calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_invocations: int = 0
    delegation_count: int = 0
    progress_events: int = 0
    no_progress_iterations: int = 0
    context_tokens_max: int = 0


@dataclass
class EnvelopeState:
    engagement_id: str
    task_id: str
    request_class: str
    profile_name: str
    role: str
    status: str
    objective: str
    selected_agents: list[str]
    qa_reserve: dict[str, int]
    budget: dict[str, int | float]
    usage: EnvelopeUsage = field(default_factory=EnvelopeUsage)
    artifact_references: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    threshold_events: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["usage_summary"] = asdict(self.usage)
        return data


@dataclass(frozen=True)
class EngagementIdentity:
    engagement_id: str
    persistence_root: Path
    engagement_path: Path


class _InterProcessLock:
    def __init__(self, path: Path, timeout_seconds: float = _LOCK_ACQUIRE_TIMEOUT_S):
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + float(self.timeout_seconds)
        payload = _json_dumps({"pid": os.getpid(), "created_at": _utc_now()}).encode("utf-8")
        while True:
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, payload)
                os.fsync(self.fd)
                return self
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Timed out waiting for governance lock: {self.path}")
                time.sleep(_LOCK_POLL_INTERVAL_S)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fd is not None:
                os.close(self.fd)
        finally:
            self.fd = None
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        return False


class GovernanceArtifactError(RuntimeError):
    """Safe, categorized failure for an existing governance artifact."""

    def __init__(self, *, artifact: str, category: str, detail: str = ""):
        self.artifact = artifact
        self.category = category
        self.fingerprint = hashlib.sha256(
            f"{artifact}:{category}:{detail}".encode("utf-8")
        ).hexdigest()
        super().__init__("invalid governance artifact")


class FileBackedBudget:
    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, dir=self.path.parent, encoding="utf-8") as fh:
                fh.write(_json_dumps(payload))
                tmp_name = fh.name
            Path(tmp_name).replace(self.path)
        except OSError as exc:
            raise GovernanceArtifactError(
                artifact=self.path.name, category="persistence_failure", detail=type(exc).__name__
            ) from exc

    def _read_existing(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except FileNotFoundError:
            raise
        except UnicodeError as exc:
            raise GovernanceArtifactError(
                artifact=self.path.name, category="encoding", detail=type(exc).__name__
            ) from exc
        except json.JSONDecodeError as exc:
            raise GovernanceArtifactError(
                artifact=self.path.name, category="invalid_json", detail=type(exc).__name__
            ) from exc
        except OSError as exc:
            raise GovernanceArtifactError(
                artifact=self.path.name, category="read_failure", detail=type(exc).__name__
            ) from exc
        if not isinstance(data, dict) or not data:
            raise GovernanceArtifactError(
                artifact=self.path.name,
                category="invalid_structure",
                detail=type(data).__name__,
            )
        required = {
            "engagement_id", "budget", "qa_reserve", "consumed", "children",
            "reservations", "request_state", "logical_calls", "budget_authorizations",
            "accounting_latch",
        }
        missing = sorted(required - set(data))
        if missing:
            raise GovernanceArtifactError(
                artifact=self.path.name, category="missing_required_fields", detail=",".join(missing)
            )
        if not isinstance(data["engagement_id"], str) or not data["engagement_id"].strip():
            raise GovernanceArtifactError(
                artifact=self.path.name, category="invalid_field_type", detail="engagement_id"
            )
        for key in ("budget", "qa_reserve", "consumed", "children", "reservations", "request_state", "logical_calls", "budget_authorizations", "accounting_latch"):
            if not isinstance(data[key], dict):
                raise GovernanceArtifactError(
                    artifact=self.path.name, category="invalid_field_type", detail=key
                )
        for auth_id, record in data["budget_authorizations"].items():
            if not isinstance(auth_id, str) or not isinstance(record, dict):
                raise GovernanceArtifactError(
                    artifact=self.path.name, category="malformed_authorization", detail="record"
                )
            required_auth = {"authorization_id", "status", "engagement_id", "profile", "task_id", "expires_at"}
            if not required_auth.issubset(record) or record.get("authorization_id") != auth_id:
                raise GovernanceArtifactError(
                    artifact=self.path.name, category="malformed_authorization", detail="required_fields"
                )
            if not isinstance(record.get("status"), str) or not isinstance(record.get("expires_at"), str):
                raise GovernanceArtifactError(
                    artifact=self.path.name, category="malformed_authorization", detail="field_type"
                )
            if record["status"] == "authorized" and not isinstance(record.get("approved_envelope"), dict):
                raise GovernanceArtifactError(
                    artifact=self.path.name, category="malformed_authorization", detail="approved_envelope"
                )
        return data

    def initialize(self, payload: dict[str, Any]) -> dict[str, Any]:
        with _InterProcessLock(self.lock_path):
            if not self.path.exists():
                self._atomic_write(payload)
                return payload
            return self._read_existing()

    def update(self, mutator):
        with _InterProcessLock(self.lock_path):
            data = self._read_existing()
            new_data = mutator(data) or data
            self._atomic_write(new_data)
            return new_data


class GovernanceController:
    _ENGAGEMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    _DECISION_ACTIONS = {"allow", "informational", "warning", "approval", "hard_stop", "disabled"}

    def __init__(self, agent: Any, config: GovernanceConfig):
        self.agent = agent
        self.config = config
        self.mode = config.mode
        self.profile_name = self._discover_profile_name()
        self.role = self._discover_role()
        self.request_class = getattr(agent, "_governance_request_class", "conversational")
        self.profile_key = getattr(agent, "_governance_profile_key", config.default_profile)
        self.identity = self.resolve_or_create_engagement_identity()
        self.root_dir = self.identity.persistence_root
        self.engagement_id = self.identity.engagement_id
        self.task_id = getattr(agent, "session_id", None) or uuid.uuid4().hex[:12]
        self.state: EnvelopeState | None = None
        self._budget_file: FileBackedBudget | None = None
        self._context_manifest: list[str] = []
        self._equivalent_tool_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}
        self._tools_used: set[str] = set()
        self._tool_schema_metrics: dict[str, Any] = {
            "total_available": 0,
            "definitions_sent": 0,
            "serialized_schema_size": 0,
            "tools_included": [],
            "unused_tools_included": 0,
        }
        self._turn_started_at = time.time()
        self._initialized = False
        self._disabled_status_path: str | None = None
        self._dispatch_failure_latched = False
        self._artifact_failure: GovernanceArtifactError | None = None
        setattr(self.agent, "_cost_context_governance_mode", self.mode)

    def _discover_profile_name(self) -> str:
        env_profile = os.environ.get("HERMES_PROFILE") or os.environ.get("HERMES_ACTIVE_PROFILE")
        if env_profile:
            return env_profile
        config_path = os.environ.get("HERMES_CONFIG_PATH", "")
        m = re.search(r"[\\/]profiles[\\/]([^\\/]+)[\\/]config\.yaml$", config_path)
        if m:
            return m.group(1)
        return "default"

    def _discover_role(self) -> str:
        seed = getattr(self.agent, "_governance_seed", None)
        if isinstance(seed, dict):
            trusted_role = seed.get("governance_role")
            if seed.get("governance_role_trusted") is True and trusted_role in {"chief", "sme", "qa"}:
                return str(trusted_role)
        role = getattr(self.agent, "_governance_role", None)
        if role and str(role) in {"chief", "sme", "qa"}:
            profile_name = (self.profile_name or "").strip().lower()
            if str(role) != "qa" or profile_name in {"security-qa", "qa"}:
                return str(role)
        if getattr(self.agent, "_delegate_depth", 0) == 0:
            return "chief"
        return "sme"

    def _is_trusted_qa_actor(self) -> bool:
        if self.role != "qa":
            return False
        seed = getattr(self.agent, "_governance_seed", None)
        if isinstance(seed, dict) and seed.get("governance_role_trusted") is True and seed.get("governance_role") == "qa":
            return True
        profile_name = (self.profile_name or "").strip().lower()
        return profile_name in {"security-qa", "qa"}

    def _resolve_workspace_dir(self) -> Path:
        cfg_path = os.environ.get("HERMES_CONFIG_PATH")
        if cfg_path and ("/profiles/" in cfg_path or "\\profiles\\" in cfg_path):
            profile_dir = Path(cfg_path).resolve().parent
            return profile_dir / str(self.config.workspace_dir)
        try:
            from hermes_constants import get_default_hermes_root, get_hermes_home

            hermes_home = get_hermes_home().resolve()
            default_root = get_default_hermes_root().resolve()
            profile_name = (self.profile_name or "").strip()
            if profile_name and hermes_home.parent.name == "profiles":
                return hermes_home.parent / profile_name / str(self.config.workspace_dir)
            if profile_name and (default_root / "profiles" / profile_name).exists():
                return default_root / "profiles" / profile_name / str(self.config.workspace_dir)
            return hermes_home / str(self.config.workspace_dir)
        except Exception:
            home = Path(os.path.expanduser("~/.hermes"))
            return home / str(self.config.workspace_dir)

    def _raw_engagement_id_candidate(self) -> Optional[str]:
        seed = getattr(self.agent, "_governance_seed", None)
        if isinstance(seed, dict):
            seeded = seed.get("engagement_id")
            if seeded is not None:
                return seeded
        return getattr(self.agent, "_governance_engagement_id", None)

    def _generate_engagement_id(self) -> str:
        return f"eng-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    def _validate_engagement_id(self, engagement_id: str) -> str:
        text = str(engagement_id or "").strip()
        if not text:
            raise ValueError("engagement_id must be a non-empty string")
        if text.lower() in {"none", "null", "undefined"}:
            raise ValueError(f"invalid engagement_id sentinel: {text!r}")
        if not self._ENGAGEMENT_ID_RE.fullmatch(text):
            raise ValueError(f"engagement_id has invalid format: {text!r}")
        return text

    def _validate_engagement_path(self, path: Path, persistence_root: Path) -> Path:
        resolved_root = persistence_root.resolve()
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(resolved_root)
        except Exception as exc:
            raise ValueError(f"engagement path escapes persistence root: {resolved_path}") from exc
        return resolved_path

    def resolve_or_create_engagement_identity(self) -> EngagementIdentity:
        raw_id = self._raw_engagement_id_candidate()
        normalized = _normalize_optional_identifier(raw_id)
        engagement_id = self._validate_engagement_id(normalized or self._generate_engagement_id())
        persistence_root = self._resolve_workspace_dir().resolve()
        engagement_path = self._validate_engagement_path(persistence_root / engagement_id, persistence_root)
        identity = EngagementIdentity(
            engagement_id=engagement_id,
            persistence_root=persistence_root,
            engagement_path=engagement_path,
        )
        self.agent._governance_engagement_id = identity.engagement_id
        return identity

    def _engagement_dir(self) -> Path:
        engagement_id = self._validate_engagement_id(self.identity.engagement_id)
        path = self._validate_engagement_path(self.identity.engagement_path, self.identity.persistence_root)
        if path.name != engagement_id:
            raise ValueError(f"engagement path/name mismatch: {path.name!r} != {engagement_id!r}")
        return path

    def _brief_path(self) -> Path:
        return self._engagement_dir() / "brief.json"

    def _record_artifact_failure(self, exc: GovernanceArtifactError, *, phase: str) -> None:
        self._artifact_failure = exc
        payload = {
            "event": "governance_artifact_invalid",
            "category": exc.category,
            "fingerprint": exc.fingerprint,
            "phase": phase,
            "artifact": exc.artifact,
            "engagement_id": self.engagement_id,
            "task_id": self.task_id,
        }
        try:
            self._append_jsonl("governance_artifact_errors.jsonl", payload)
        except Exception as persist_exc:
            self._dispatch_failure_latched = True
            self._artifact_failure = GovernanceArtifactError(
                artifact=exc.artifact,
                category="persistence_failure",
                detail=type(persist_exc).__name__,
            )

    def _read_json(self, path: Path, default: Optional[dict] = None) -> dict[str, Any]:
        if not path.exists():
            return dict(default or {})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return dict(default or {})
        return data if isinstance(data, dict) else dict(default or {})

    def _persist_json(self, name: str, payload: dict) -> str:
        path = self._engagement_dir() / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json_dumps(payload), encoding="utf-8")
        return str(path)

    def _append_jsonl(self, name: str, payload: dict) -> str:
        path = self._engagement_dir() / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(_json_dumps(payload) + "\n")
        return str(path)

    def _write_mode_status(self, *, active: bool, reason: str = "") -> str:
        payload = {
            "timestamp": _utc_now(),
            "mode": self.mode,
            "active": bool(active),
            "reason": reason,
            "profile": self.profile_name,
            "engagement_id": self.engagement_id,
            "task_id": self.task_id,
        }
        path = self._persist_json("governance_status.json", payload)
        setattr(self.agent, "_cost_context_governance_status_path", path)
        return path

    def is_active(self) -> bool:
        return self.mode in {"observe", "enforce"}

    def should_enforce(self) -> bool:
        return self.mode == "enforce"

    def _profile_budget(self) -> BudgetProfile:
        return self.config.profiles.get(self.profile_key, self.config.profiles[self.config.default_profile])

    def _brief_preservation_snapshot(self) -> dict[str, Any]:
        brief = self._read_json(self._brief_path(), default={})
        return {key: brief.get(key, []) for key in _BRIEF_PRESERVE_KEYS}

    def classify(self, user_message: str) -> tuple[str, str]:
        text = (user_message or "").lower()
        force = self.profile_name in set(self.config.force_for_profiles)
        if getattr(self.agent, "_delegate_depth", 0) > 0:
            return "engagement", getattr(self.agent, "_governance_profile_key", "bounded")
        if any(word in text for word in self.config.engagement_keywords):
            return "engagement", "standard_engagement"
        if force and ("security" in text or "miniciso" in text):
            return "engagement", "standard_engagement"
        if any(marker in text for marker in ["review", "assessment", "investigate", "analyze"]):
            return "bounded", "bounded"
        return "conversational", "conversational"

    def begin_turn(self, *, user_message: str, system_message: str, messages: list[dict], task_id: str | None = None) -> None:
        self._turn_started_at = time.time()
        if self.mode == "disabled":
            self.engagement_id = self.engagement_id or f"disabled-{uuid.uuid4().hex[:8]}"
            self.task_id = task_id or self.task_id
            self._disabled_status_path = self._write_mode_status(active=False, reason="operator_disabled")
            return
        seed = getattr(self.agent, "_governance_seed", None)
        self.request_class, self.profile_key = self.classify(user_message)
        if getattr(self.agent, "_governance_profile_key", None):
            self.profile_key = self.agent._governance_profile_key
        if getattr(self.agent, "_governance_request_class", None):
            self.request_class = self.agent._governance_request_class
        if isinstance(seed, dict) and _normalize_optional_identifier(seed.get("engagement_id")):
            self.profile_key = str(seed.get("profile_key") or self.profile_key)
            self.request_class = str(seed.get("request_class") or self.request_class)
        self.identity = self.resolve_or_create_engagement_identity()
        self.root_dir = self.identity.persistence_root
        self.engagement_id = self.identity.engagement_id
        self.task_id = task_id or self.task_id
        budget = self._profile_budget()
        if isinstance(seed, dict) and isinstance(seed.get("budget_override"), dict):
            budget = BudgetProfile.from_mapping(seed.get("budget_override"), fallback=budget)
        qa_tokens = int(budget.total_tokens * float(budget.qa_reserve_ratio or 0.0))
        qa_calls = int(budget.total_model_calls * float(budget.qa_reserve_ratio or 0.0))
        self.state = EnvelopeState(
            engagement_id=self.engagement_id,
            task_id=self.task_id,
            request_class=self.request_class,
            profile_name=self.profile_key,
            role=self.role,
            status="active",
            objective="",
            selected_agents=list(seed.get("selected_agents") or []) if isinstance(seed, dict) else [],
            qa_reserve={"total_tokens": qa_tokens, "model_calls": qa_calls},
            budget=asdict(budget),
        )
        self._budget_file = FileBackedBudget(self._engagement_dir() / "budget.json")
        if not self._initialized:
            try:
                self._budget_file.initialize(self._bootstrap_budget({}, budget))
            except GovernanceArtifactError as exc:
                self._record_artifact_failure(exc, phase="begin_turn")
                return
            self._initialized = True
        brief = self._read_json(self._brief_path())
        brief.setdefault("scope", [])
        brief.setdefault("scope_exclusions", [])
        brief.setdefault("decisions", [])
        brief.setdefault("constraints", [])
        brief.setdefault("qa_obligations", [])
        brief.setdefault("selected_agents", list(seed.get("selected_agents") or []) if isinstance(seed, dict) else [])
        brief.setdefault("artifact_references", [])
        brief.setdefault("evidence_ids", [])
        brief.setdefault("claim_ids", [])
        brief.setdefault("evidence_ledger", "evidence_ledger.jsonl")
        brief.setdefault("claim_ledger", "claim_ledger.jsonl")
        brief.setdefault("open_questions", [])
        brief.setdefault("checkpoints", [])
        brief.update({
            "engagement_id": self.engagement_id,
            "request_class": self.request_class,
            "objective": brief.get("objective") or _text_metadata(user_message),
            "resource_envelope": asdict(budget),
            "qa_reserve": self.state.qa_reserve,
            "status": self.state.status,
            "usage_summary": asdict(self.state.usage),
        })
        self._persist_json("brief.json", brief)
        self.record_context_manifest(messages, system_message=system_message)
        self.log_event("turn_begin", {
            "request_class": self.request_class,
            "profile": self.profile_key,
            "session_id": getattr(self.agent, "session_id", None),
            "agent_role": self.role,
            "selected_agents": self.state.selected_agents,
            "mode": self.mode,
        })
        self._write_mode_status(active=True, reason="governance_active")
        self.agent._governance_engagement_id = self.engagement_id
        self.agent._governance_request_class = self.request_class
        self.agent._governance_profile_key = self.profile_key
        self.agent._governance_role = self.role

    def _bootstrap_budget(self, data: dict, budget: BudgetProfile) -> dict:
        if data.get("engagement_id"):
            self._refresh_budget_pools(data)
            return data
        bootstrap = {
            "engagement_id": self.engagement_id,
            "budget": asdict(budget),
            "qa_reserve": {
                "total_tokens": int(budget.total_tokens * float(budget.qa_reserve_ratio or 0.0)),
                "model_calls": int(budget.total_model_calls * float(budget.qa_reserve_ratio or 0.0)),
            },
            "consumed": asdict(EnvelopeUsage()),
            "children": {},
            "reservations": {},
            "request_state": {},
            "logical_calls": {},
            "budget_authorizations": {},
            "accounting_latch": {
                "active": False,
                "reason": None,
                "request_ids": [],
                "updated_at": _utc_now(),
            },
        }
        self._refresh_budget_pools(bootstrap)
        return bootstrap

    def _decision_payload(
        self,
        *,
        request_id: str,
        action: str,
        allowed: bool,
        reason: str,
        compact_context: bool = False,
        termination_reason: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            **extra,
            "request_id": request_id,
            "allowed": bool(allowed),
            "action": action,
            "reason": reason,
            "compact_context": bool(compact_context),
            "termination_reason": termination_reason,
        }

    def validate_decision(self, decision: Any, *, request_id: str) -> dict[str, Any]:
        """Validate the provider-admission contract at its trust boundary."""
        error = None
        if not isinstance(decision, dict):
            error = "decision must be a mapping"
        else:
            required = {"allowed", "action", "reason", "compact_context", "termination_reason"}
            missing = sorted(required - set(decision))
            if missing:
                error = f"missing fields: {', '.join(missing)}"
            elif type(decision.get("allowed")) is not bool:
                error = "allowed must be bool"
            elif str(decision.get("action")) not in self._DECISION_ACTIONS:
                error = f"unknown action: {decision.get('action')!r}"
            elif not isinstance(decision.get("reason"), str) or not decision.get("reason", "").strip():
                error = "reason must be a non-empty string"
            elif type(decision.get("compact_context")) is not bool:
                error = "compact_context must be bool"
            elif decision.get("termination_reason") is not None and not isinstance(decision.get("termination_reason"), str):
                error = "termination_reason must be string or null"
        if error is None:
            normalized = dict(decision)
            normalized["request_id"] = str(normalized.get("request_id") or request_id)
            return normalized
        if not self.should_enforce():
            return self._decision_payload(
                request_id=request_id,
                action="informational" if self.mode == "observe" else "disabled",
                allowed=True,
                reason="invalid_decision_observed" if self.mode == "observe" else "operator_disabled",
                validation_error=error,
            )
        raise RuntimeError(f"invalid governance decision: {error}")

    def _validate_authorization_envelope(self, envelope: Any) -> dict[str, Any]:
        if not isinstance(envelope, dict):
            raise RuntimeError("authorization envelope must be explicit")
        operation = str(envelope.get("operation") or "")
        if operation not in {"increment", "replace"}:
            raise RuntimeError("authorization envelope operation must be increment or replace")
        normalized: dict[str, Any] = {"operation": operation}
        for key in ("total_tokens", "total_model_calls", "context_tokens_per_request"):
            value = envelope.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeError(f"authorization envelope {key} must be a positive integer")
            normalized[key] = int(value)
        return normalized

    def _authorization_owner(self) -> dict[str, str]:
        subject = str(getattr(self.agent, "_user_id", None) or "").strip()
        platform = str(getattr(self.agent, "platform", None) or "").strip()
        chat_id = str(getattr(self.agent, "_chat_id", None) or "").strip()
        trusted = getattr(self.agent, "_governance_owner_identity_trusted", False) is True
        if not trusted or not subject or not platform or not chat_id:
            return {}
        return {"subject": subject, "platform": platform, "profile": self.profile_name, "chat_id": chat_id}

    def _authorization_audit(self, event: str, record: dict[str, Any], **extra: Any) -> None:
        payload = {
            "timestamp": _utc_now(),
            "event": event,
            "authorization_id": record.get("authorization_id"),
            "engagement_id": self.engagement_id,
            "profile": self.profile_name,
            "blocked_request_id": record.get("blocked_request_id"),
            "logical_call_id": record.get("logical_call_id"),
            "status": record.get("status"),
            **extra,
        }
        self._append_jsonl("authorization_audit.jsonl", payload)
        self.log_event("budget_authorization", payload)

    def create_budget_authorization(
        self,
        *,
        blocked_request_id: str,
        logical_call_id: str | None,
        decision_action: str,
        proposed_envelope: dict[str, Any],
        ttl_seconds: int = 900,
    ) -> dict[str, Any]:
        if self._budget_file is None or not self.should_enforce():
            raise RuntimeError("budget authorization requires active enforce governance")
        if decision_action not in {"approval", "hard_stop"}:
            raise RuntimeError("budget authorization requires approval or hard_stop")
        envelope = self._validate_authorization_envelope(proposed_envelope)
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or not 1 <= ttl_seconds <= 3600:
            raise RuntimeError("authorization TTL must be between 1 and 3600 seconds")
        owner = self._authorization_owner()
        authorization_id = f"auth-{uuid.uuid4().hex}"
        nonce = uuid.uuid4().hex + uuid.uuid4().hex
        callback_token = secrets.token_urlsafe(32)
        created_at = datetime.now(timezone.utc)
        record = {
            "authorization_id": authorization_id,
            "nonce_hash": hashlib.sha256(nonce.encode("utf-8")).hexdigest(),
            "callback_hash": hashlib.sha256(callback_token.encode("utf-8")).hexdigest(),
            "status": "pending",
            "engagement_id": self.engagement_id,
            "profile": self.profile_name,
            "task_id": self.task_id,
            "blocked_request_id": blocked_request_id,
            "logical_call_id": logical_call_id,
            "decision_action": decision_action,
            "proposed_envelope": envelope,
            "approved_envelope": None,
            "owner": owner,
            "authorized_by": None,
            "created_at": created_at.isoformat(),
            "expires_at": (created_at + timedelta(seconds=ttl_seconds)).isoformat(),
            "consumed_at": None,
            "consumed_by_request_id": None,
        }

        def mutate(data: dict) -> dict:
            authorizations = data.setdefault("budget_authorizations", {})
            for existing in authorizations.values():
                if (
                    isinstance(existing, dict)
                    and existing.get("logical_call_id") == logical_call_id
                    and existing.get("status") in {"pending", "authorized"}
                ):
                    existing["status"] = "superseded"
                    existing["superseded_at"] = _utc_now()
                    existing["superseded_by"] = authorization_id
            authorizations[authorization_id] = record
            return data

        self._budget_file.update(mutate)
        self._authorization_audit("authorization_requested", record)
        instructions = (
            "Use only the authenticated callback to submit this Telegram decision; "
            f"it expires at {record['expires_at']} and carries the server-side envelope. "
            "Ordinary chat messages are never authorization."
        )
        return {
            "authorization_id": authorization_id,
            "nonce": nonce,
            "callback_token": callback_token,
            "callback_data": {
                "approve": f"ba:{callback_token}:a",
                "deny": f"ba:{callback_token}:d",
            },
            "expires_at": record["expires_at"],
            "engagement_id": self.engagement_id,
            "profile": self.profile_name,
            "blocked_request_id": blocked_request_id,
            "logical_call_id": logical_call_id,
            "proposed_envelope": envelope,
            "instructions": instructions,
        }

    def submit_budget_authorization(
        self,
        *,
        authorization_id: str | None = None,
        nonce: str | None = None,
        callback_token: str | None = None,
        callback_capability: Any,
        principal: dict[str, Any],
        approved: bool,
        envelope: dict[str, Any] | None,
        reason: str,
    ) -> dict[str, Any]:
        if self._budget_file is None or not self.should_enforce():
            raise RuntimeError("budget authorization callback requires active enforce governance")
        expected_capability = getattr(self.agent, "_governance_callback_capability", None)
        if expected_capability is None or callback_capability is not expected_capability:
            raise RuntimeError("authenticated callback capability required")
        if type(approved) is not bool:
            raise RuntimeError("budget authorization decision must be bool")
        if not isinstance(reason, str) or not reason.strip():
            raise RuntimeError("budget authorization reason is required")
        if callback_token is None and (not authorization_id or nonce is None):
            raise RuntimeError("authorization callback reference required")
        if callback_token is None and envelope is None:
            raise RuntimeError("authorization envelope required")
        normalized_envelope = self._validate_authorization_envelope(envelope) if envelope is not None else None
        outcome: dict[str, Any] = {}

        def mutate(data: dict) -> dict:
            nonlocal outcome, normalized_envelope
            authorizations = data.setdefault("budget_authorizations", {})
            resolved_authorization_id = authorization_id
            if callback_token is not None:
                callback_hash = hashlib.sha256(callback_token.encode("utf-8")).hexdigest()
                resolved_authorization_id = next(
                    (
                        key for key, candidate in authorizations.items()
                        if isinstance(candidate, dict)
                        and candidate.get("status") == "pending"
                        and hmac.compare_digest(str(candidate.get("callback_hash") or ""), callback_hash)
                    ),
                    None,
                )
            record = authorizations.get(resolved_authorization_id)
            if not isinstance(record, dict):
                raise RuntimeError("unknown budget authorization")
            status = str(record.get("status") or "")
            if status == "consumed":
                raise RuntimeError("budget authorization already consumed")
            if status != "pending":
                raise RuntimeError(f"budget authorization is {status or 'invalid'}")
            expires_at = datetime.fromisoformat(str(record.get("expires_at")))
            if datetime.now(timezone.utc) >= expires_at:
                record["status"] = "expired"
                record["expired_at"] = _utc_now()
                outcome = dict(record)
                return data
            expected_hash = str(record.get("nonce_hash") or "")
            if callback_token is None:
                actual_hash = hashlib.sha256(str(nonce or "").encode("utf-8")).hexdigest()
                if not expected_hash or not hmac.compare_digest(expected_hash, actual_hash):
                    raise RuntimeError("budget authorization nonce mismatch")
            if not isinstance(principal, dict) or principal.get("authenticated") is not True:
                raise RuntimeError("authenticated principal required")
            if principal.get("authentication_source") != "gateway_adapter":
                raise RuntimeError("authenticated gateway adapter source required")
            if str(principal.get("profile") or "") != str(record.get("profile") or ""):
                raise RuntimeError("budget authorization profile mismatch")
            if str(principal.get("engagement_id") or "") != str(record.get("engagement_id") or ""):
                raise RuntimeError("budget authorization engagement mismatch")
            owner = record.get("owner") or {}
            if not owner or str(principal.get("subject") or "") != str(owner.get("subject") or ""):
                raise RuntimeError("budget authorization owner mismatch")
            if str(principal.get("platform") or "") != str(owner.get("platform") or ""):
                raise RuntimeError("budget authorization platform mismatch")
            if str(principal.get("chat_id") or "") != str(owner.get("chat_id") or ""):
                raise RuntimeError("budget authorization chat mismatch")
            if str(principal.get("task_id") or "") != str(record.get("task_id") or ""):
                raise RuntimeError("budget authorization task mismatch")
            record["status"] = "authorized" if approved else "denied"
            if approved and normalized_envelope is None:
                normalized_envelope = self._validate_authorization_envelope(record.get("proposed_envelope"))
            record["approved_envelope"] = normalized_envelope if approved else None
            record["decision_reason"] = reason.strip()
            record["authorized_by"] = {
                "subject": str(principal.get("subject")),
                "platform": str(principal.get("platform")),
                "profile": str(principal.get("profile")),
                "authentication_source": "gateway_adapter",
            }
            record["decided_at"] = _utc_now()
            outcome = dict(record)
            return data

        self._budget_file.update(mutate)
        if outcome.get("status") == "expired":
            self._authorization_audit("authorization_expired", outcome)
            raise RuntimeError("budget authorization expired")
        self._authorization_audit(
            "authorization_authorized" if approved else "authorization_denied",
            outcome,
            authorized_by=outcome.get("authorized_by"),
            approved_envelope=outcome.get("approved_envelope"),
        )
        if approved:
            self.agent._governance_resume_authorization_id = outcome.get("authorization_id")
            self.agent._governance_resume_logical_call_id = outcome.get("logical_call_id")
            self.agent._governance_resume_task_id = outcome.get("task_id")
        return outcome

    def submit_budget_authorization_callback(
        self,
        *,
        callback_token: str,
        principal: dict[str, Any],
        approved: bool,
        reason: str,
    ) -> dict[str, Any]:
        """Consume a Telegram callback token without exposing request payloads."""
        return self.submit_budget_authorization(
            callback_token=callback_token,
            callback_capability=getattr(self.agent, "_governance_callback_capability", None),
            principal=principal,
            approved=approved,
            envelope=None,
            reason=reason,
        )

    def _consume_budget_authorization(
        self,
        *,
        request_id: str,
        logical_call_id: str | None,
        approx_request_tokens: int,
    ) -> dict[str, Any] | None:
        if self._budget_file is None:
            return None
        resume_authorization_id = str(
            getattr(self.agent, "_governance_resume_authorization_id", None) or ""
        ).strip()
        if not resume_authorization_id:
            return None
        consumed: dict[str, Any] | None = None

        def mutate(data: dict) -> dict:
            nonlocal consumed
            authorizations = data.setdefault("budget_authorizations", {})
            record = authorizations.get(resume_authorization_id)
            if not isinstance(record, dict) or record.get("status") != "authorized":
                return data
            if record.get("engagement_id") != self.engagement_id or record.get("profile") != self.profile_name:
                raise RuntimeError("budget authorization scope mismatch on resume")
            if record.get("task_id") != self.task_id:
                raise RuntimeError("budget authorization task mismatch on resume")
            if record.get("logical_call_id") != logical_call_id:
                return data
            expires_at = datetime.fromisoformat(str(record.get("expires_at")))
            if datetime.now(timezone.utc) >= expires_at:
                record["status"] = "expired"
                record["expired_at"] = _utc_now()
                return data
            if request_id == record.get("blocked_request_id"):
                raise RuntimeError("budget authorization requires a new request_id")
            envelope = self._validate_authorization_envelope(record.get("approved_envelope"))
            if approx_request_tokens > int(envelope["context_tokens_per_request"]):
                raise RuntimeError("authorized per-request envelope is insufficient")
            record["status"] = "consumed"
            record["consumed_at"] = _utc_now()
            record["consumed_by_request_id"] = request_id
            record["consumed_by_logical_call_id"] = logical_call_id
            entry = self._request_state_entry(data, request_id)
            entry["budget_authorization"] = {
                "authorization_id": record["authorization_id"],
                "envelope": envelope,
            }
            consumed = dict(record)
            return data

        self._budget_file.update(mutate)
        if consumed:
            self.agent._governance_resume_authorization_id = None
            self.agent._governance_resume_logical_call_id = None
            self._authorization_audit("authorization_consumed", consumed, consumed_by_request_id=request_id)
        return consumed

    def log_event(self, event: str, payload: dict[str, Any]) -> None:
        if not self.is_active():
            return
        record = {
            "timestamp": _utc_now(),
            "event": event,
            "engagement_id": self.engagement_id,
            "task_id": self.task_id,
            "session_id": getattr(self.agent, "session_id", None),
            "agent_role": self.role,
            **payload,
        }
        self._append_jsonl("telemetry.jsonl", project_artifact_value(record))

    def _request_trace_enabled(self) -> bool:
        value = os.getenv("HERMES_GOVERNANCE_TRACE_REQUESTS", "")
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _sanitized_stack_summary(self, *, limit: int = 6) -> list[str]:
        frames = []
        for frame in traceback.extract_stack(limit=18)[:-2]:
            filename = Path(frame.filename).name
            frames.append(f"{filename}:{frame.name}:{frame.lineno}")
        return frames[-limit:]

    def _request_snapshot_view(self, snapshot: dict[str, Any], request_id: str) -> dict[str, Any]:
        request_state = ((snapshot.get("request_state") or {}).get(request_id) or {}) if isinstance(snapshot, dict) else {}
        reservations = (snapshot.get("reservations") or {}) if isinstance(snapshot, dict) else {}
        consumed = (snapshot.get("consumed") or {}) if isinstance(snapshot, dict) else {}
        return {
            "request_state": str(request_state.get("state") or "created"),
            "reservation_id": request_state.get("reservation_id") or (request_id if request_id in reservations else None),
            "reservation_estimate": int(request_state.get("reservation_estimate", 0) or 0),
            "consumed_total_tokens": int(consumed.get("total_tokens", 0) or 0),
            "consumed_model_calls": int(consumed.get("model_calls", 0) or 0),
            "open_reservations": sorted(list(reservations.keys())),
            "open_reservation_count": len(reservations),
        }

    def _log_request_transition(self, payload: dict[str, Any]) -> None:
        if self._request_trace_enabled():
            request_id = str(payload.get("request_id") or "")
            current = self._request_snapshot_view(self._root_snapshot(), request_id) if request_id else {}
            seed = getattr(self.agent, "_governance_seed", None)
            trusted_role = bool(getattr(self.agent, "_governance_role_trusted", False))
            if isinstance(seed, dict) and seed.get("governance_role_trusted") is True:
                trusted_role = True
            stack_summary = self._sanitized_stack_summary()
            event = {
                "trace_event_id": uuid.uuid4().hex[:12],
                "controller_instance_id": hex(id(self)),
                "agent_profile": getattr(self.agent, "profile", None),
                "agent_role": self.role,
                "trusted_role": trusted_role,
                "engagement_id": self.engagement_id,
                "request_id": request_id,
                "phase": payload.get("phase") or payload.get("transition"),
                "call_site": stack_summary[-1] if stack_summary else None,
                "stack_summary": stack_summary,
                "state_before": payload.get("state_before"),
                "state_after": payload.get("state_after", current.get("request_state")),
                "decision": payload.get("decision"),
                "termination_reason": payload.get("termination_reason"),
                "reservation_id": payload.get("reservation_id", current.get("reservation_id")),
                "reservation_estimate": payload.get("reservation_estimate", current.get("reservation_estimate")),
                "consumed_before": payload.get("consumed_before"),
                "consumed_after": payload.get("consumed_after", {"total_tokens": current.get("consumed_total_tokens"), "model_calls": current.get("consumed_model_calls")}),
                "open_reservation_ids": payload.get("outstanding_reservations", current.get("open_reservations", [])),
                "open_reservation_count": len(payload.get("outstanding_reservations", current.get("open_reservations", [])) or []),
                "controller_phase": payload.get("transition"),
            }
            self.log_event("request_trace", event)

    def record_context_manifest(self, messages: list[dict], *, system_message: str = "") -> list[str]:
        blocks = []
        if system_message:
            blocks.append({"role": "system", "content": system_message[:2000]})
        for msg in messages[-8:]:
            blocks.append({"role": msg.get("role"), "content": str(msg.get("content", ""))[:2000]})
        self._context_manifest = [hash_context_block(block) for block in blocks]
        self.log_event("context_manifest", {
            "context_manifest": self._context_manifest,
            "context_block_count": len(blocks),
            "context_tokens_estimated": estimate_messages_tokens(blocks),
        })
        return list(self._context_manifest)

    def filter_tool_schemas(
        self,
        tool_defs: list[dict],
        *,
        total_available: Optional[list[dict]] = None,
        required_tools: Optional[Iterable[str]] = None,
    ) -> list[dict]:
        required = set(required_tools or [])
        filtered = list(tool_defs or [])
        if required:
            names = {item.get("function", {}).get("name") for item in filtered}
            missing = sorted(required - names)
            if missing:
                raise RuntimeError(f"Required tools missing after governance filtering: {missing}")
        self.record_tool_schema_metrics(
            filtered,
            [],
            total_available_count=len(total_available or tool_defs or []),
        )
        return filtered

    def record_tool_schema_metrics(
        self,
        tool_defs: list[dict],
        tools_used: Optional[Iterable[str]] = None,
        *,
        total_available_count: Optional[int] = None,
    ) -> None:
        serialized = _json_dumps(tool_defs or [])
        used = set(tools_used or [])
        included_names = [item.get("function", {}).get("name") for item in tool_defs or []]
        included_name_set = {name for name in included_names if name}
        count = len(tool_defs or [])
        unused = max(0, count - len(used & included_name_set))
        self._tool_schema_metrics = {
            "total_available": int(total_available_count if total_available_count is not None else count),
            "definitions_sent": count,
            "serialized_schema_size": len(serialized.encode("utf-8")),
            "tools_included": included_names,
            "unused_tools_included": unused,
            "tools_used": sorted(used),
        }
        self.log_event("tool_schema", {
            "tool_definition_count": count,
            "tool_definition_total_available": self._tool_schema_metrics["total_available"],
            "tool_schema_serialized_size": self._tool_schema_metrics["serialized_schema_size"],
            "tools_included": included_names,
            "tools_actually_invoked": sorted(used),
            "unused_tools_included": unused,
        })

    def _consume(self, **delta: int) -> dict:
        budget_file = self._budget_file
        if budget_file is None:
            return {}

        def mutate(data: dict) -> dict:
            consumed = data.setdefault("consumed", asdict(EnvelopeUsage()))
            children = data.setdefault("children", {})
            child = children.setdefault(self.task_id, asdict(EnvelopeUsage()))
            for key, value in delta.items():
                consumed[key] = int(consumed.get(key, 0)) + int(value)
                child[key] = int(child.get(key, 0)) + int(value)
            child["updated_at"] = _utc_now()
            return data

        updated = budget_file.update(mutate)
        if self.state is not None:
            for key, value in delta.items():
                if hasattr(self.state.usage, key):
                    setattr(self.state.usage, key, int(getattr(self.state.usage, key)) + int(value))
            self.state.updated_at = _utc_now()
        return updated

    def _root_snapshot(self) -> dict[str, Any]:
        if self._budget_file is None:
            return {}
        return self._budget_file.update(lambda current: current)

    def _root_consumed(self) -> dict[str, int]:
        return dict(self._root_snapshot().get("consumed", {}))

    def _refresh_budget_pools(self, data: dict) -> dict:
        budget_data = data.get("budget", {}) or asdict(self._profile_budget())
        reserve = data.get("qa_reserve", {}) or {}
        consumed = data.get("consumed", {}) or {}
        reservations = data.get("reservations", {}) or {}
        reserved_tokens = sum(int((item or {}).get("total_tokens", 0)) for item in reservations.values())
        total_tokens = int(budget_data.get("total_tokens", self._profile_budget().total_tokens))
        qa_reserved = int(reserve.get("total_tokens", 0))
        total_consumed = int(consumed.get("total_tokens", 0)) + reserved_tokens
        qa_consumed = sum(
            int((item or {}).get("total_tokens", 0))
            for item in reservations.values()
            if (item or {}).get("agent_role") == "qa"
        )
        pools = data.setdefault("pools", {})
        pools.update(
            {
                "general_available": max(0, total_tokens - qa_reserved - total_consumed),
                "qa_reserved": qa_reserved,
                "qa_consumed": qa_consumed,
                "total_consumed": total_consumed,
            }
        )
        return data

    def _request_state_entry(self, data: dict, request_id: str) -> dict:
        request_state = data.setdefault("request_state", {})
        return request_state.setdefault(
            request_id,
            {
                "request_id": request_id,
                "task_id": self.task_id,
                "engagement_id": self.engagement_id,
                "agent_role": self.role,
                "state": "created",
                "reservation_id": None,
                "reservation_estimate": 0,
                "decision": None,
                "logical_call_id": None,
                "attempt_id": None,
                "request_kind": "primary",
                "termination_reason": None,
                "dispatch_count": 0,
                "reconciliation": None,
                "released_reason": None,
                "updated_at": _utc_now(),
            },
        )

    def _logical_call_entry(self, data: dict, logical_call_id: str) -> dict:
        logical_calls = data.setdefault("logical_calls", {})
        return logical_calls.setdefault(
            logical_call_id,
            {
                "logical_call_id": logical_call_id,
                "task_id": self.task_id,
                "engagement_id": self.engagement_id,
                "agent_role": self.role,
                "request_ids": [],
                "last_request_id": None,
                "last_attempt_id": None,
                "next_attempt_ordinal": 1,
                "last_allocated_attempt_ordinal": 0,
                "last_allocated_request_kind": None,
                "preflight_count": 0,
                "compaction_attempts": 0,
                "no_progress_count": 0,
                "termination_reason": None,
                "last_context_tokens": 0,
                "last_context_fingerprint": None,
                "updated_at": _utc_now(),
            },
        )

    def _allocate_request_identity_mutation(
        self,
        data: dict,
        *,
        logical_call_id: str,
        request_kind: str,
    ) -> dict:
        entry = self._logical_call_entry(data, logical_call_id)
        ordinal = max(1, int(entry.get("next_attempt_ordinal", 1) or 1))
        entry["last_allocated_attempt_ordinal"] = ordinal
        entry["last_allocated_request_kind"] = request_kind
        entry["next_attempt_ordinal"] = ordinal + 1
        entry["updated_at"] = _utc_now()
        return self._refresh_budget_pools(data)

    def allocate_request_identity(
        self,
        *,
        logical_call_id: str,
        request_kind: str = "primary",
        request_label: str = "request",
    ) -> dict[str, Any]:
        if self._budget_file is None:
            ordinal = 1
        else:
            snapshot = self._budget_file.update(
                lambda data: self._allocate_request_identity_mutation(
                    data,
                    logical_call_id=logical_call_id,
                    request_kind=request_kind,
                )
            )
            entry = ((snapshot.get("logical_calls") or {}).get(logical_call_id) or {}) if isinstance(snapshot, dict) else {}
            ordinal = int(entry.get("last_allocated_attempt_ordinal") or 1)
        attempt_id = f"{logical_call_id}:attempt:{ordinal}"
        if request_kind == "primary":
            request_id = f"{attempt_id}:{request_label}"
        else:
            request_id = f"{attempt_id}:{request_kind}:{request_label}"
        return {
            "logical_call_id": logical_call_id,
            "attempt_ordinal": ordinal,
            "attempt_id": attempt_id,
            "request_id": request_id,
            "request_kind": request_kind,
        }

    def _context_fingerprint(self, messages: list[dict]) -> str:
        parts: list[str] = []
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "")
            content = msg.get("content")
            text = _json_dumps(content) if isinstance(content, list) else str(content or "")
            parts.append(f"{role}:{text[:400]}")
        return hashlib.sha256("\n".join(parts).encode("utf-8", errors="ignore")).hexdigest()

    def _record_logical_compaction_result_mutation(
        self,
        data: dict,
        *,
        logical_call_id: str,
        attempt_id: str,
        request_id: str,
        pre_tokens: int,
        post_tokens: int,
        pre_fingerprint: str,
        post_fingerprint: str,
        material_progress: bool,
    ) -> dict:
        entry = self._logical_call_entry(data, logical_call_id)
        entry["compaction_attempts"] = int(entry.get("compaction_attempts", 0)) + 1
        entry["last_request_id"] = request_id
        entry["last_attempt_id"] = attempt_id
        entry["last_context_tokens"] = int(post_tokens)
        entry["last_context_fingerprint"] = post_fingerprint
        if not material_progress or pre_fingerprint == post_fingerprint:
            entry["no_progress_count"] = int(entry.get("no_progress_count", 0)) + 1
            entry["termination_reason"] = "logical_call_no_progress"
        elif int(entry.get("compaction_attempts", 0)) >= int(self.config.progress.logical_call_max_compactions):
            entry["termination_reason"] = "logical_call_preflight_limit"
        entry["updated_at"] = _utc_now()
        return self._refresh_budget_pools(data)

    def record_logical_compaction_result(
        self,
        *,
        logical_call_id: str,
        attempt_id: str,
        request_id: str,
        pre_messages: list[dict],
        post_messages: list[dict],
    ) -> dict[str, Any]:
        if self._budget_file is None:
            return {"stop": False, "request_id": request_id}

        pre_tokens = estimate_messages_tokens(pre_messages)
        post_tokens = estimate_messages_tokens(post_messages)
        pre_fingerprint = self._context_fingerprint(pre_messages)
        post_fingerprint = self._context_fingerprint(post_messages)
        saved_tokens = max(0, pre_tokens - post_tokens)
        min_saved_tokens = max(1, int(self.config.progress.logical_call_min_progress_tokens))
        min_ratio = max(0.0, float(self.config.progress.logical_call_min_progress_ratio))
        material_progress = saved_tokens >= min_saved_tokens or (
            pre_tokens > 0 and (saved_tokens / max(1, pre_tokens)) >= min_ratio
        )
        snapshot = self._budget_file.update(
            lambda data: self._record_logical_compaction_result_mutation(
                data,
                logical_call_id=logical_call_id,
                attempt_id=attempt_id,
                request_id=request_id,
                pre_tokens=pre_tokens,
                post_tokens=post_tokens,
                pre_fingerprint=pre_fingerprint,
                post_fingerprint=post_fingerprint,
                material_progress=material_progress,
            )
        )
        entry = ((snapshot.get("logical_calls") or {}).get(logical_call_id) or {}) if isinstance(snapshot, dict) else {}
        termination_reason = entry.get("termination_reason")
        payload = {
            "logical_call_id": logical_call_id,
            "attempt_id": attempt_id,
            "request_id": request_id,
            "pre_tokens": pre_tokens,
            "post_tokens": post_tokens,
            "saved_tokens": saved_tokens,
            "material_progress": material_progress,
            "termination_reason": termination_reason,
        }
        self.log_event("logical_compaction_result", payload)
        if termination_reason == "logical_call_no_progress":
            return {
                **payload,
                "stop": True,
                "message": "Governança encerrou a operação lógica antes do provider: logical_call_no_progress.",
            }
        if termination_reason == "logical_call_preflight_limit":
            return {
                **payload,
                "stop": True,
                "message": "Governança encerrou a operação lógica antes do provider: logical_call_preflight_limit.",
            }
        return {**payload, "stop": False}

    def _request_terminal_states(self) -> set[str]:
        return {
            "completed",
            "released_before_dispatch",
            "provider_error_accounted",
            "interrupted_accounted",
            "accounting_error",
            "blocked",
        }

    def _request_transition_map(self) -> dict[str, set[str]]:
        return {
            "created": {"evaluated", "reserved", "released_before_dispatch", "blocked"},
            "evaluated": {"reserved", "released_before_dispatch", "blocked"},
            "reserved": {"dispatched", "released_before_dispatch", "blocked", "accounting_error"},
            "dispatched": {"reconciled", "provider_error_accounted", "interrupted_accounted", "accounting_error"},
            "reconciled": {"completed"},
            "completed": set(),
            "released_before_dispatch": set(),
            "provider_error_accounted": set(),
            "interrupted_accounted": set(),
            "accounting_error": set(),
            "blocked": set(),
        }

    def _assert_request_transition(self, current_state: str, next_state: str, *, request_id: str) -> None:
        if current_state == next_state:
            return
        allowed = self._request_transition_map().get(current_state, set())
        if next_state not in allowed:
            raise RuntimeError(f"Invalid request state transition for {request_id}: {current_state} -> {next_state}")

    def _engagement_accounting_latch(self, data: dict | None = None) -> dict[str, Any]:
        source = data if isinstance(data, dict) else self._root_snapshot()
        latch = source.get("accounting_latch")
        if not isinstance(latch, dict):
            latch = {
                "active": False,
                "reason": None,
                "request_ids": [],
                "updated_at": _utc_now(),
            }
        return {
            "active": bool(latch.get("active", False)),
            "reason": latch.get("reason"),
            "request_ids": list(latch.get("request_ids") or []),
            "updated_at": latch.get("updated_at"),
        }

    def _activate_accounting_latch(self, data: dict, *, request_id: str, reason: str) -> dict:
        current = self._engagement_accounting_latch(data)
        request_ids = list(current.get("request_ids") or [])
        if request_id not in request_ids:
            request_ids.append(request_id)
        data["accounting_latch"] = {
            "active": True,
            "reason": reason,
            "request_ids": request_ids,
            "updated_at": _utc_now(),
        }
        return data

    def evaluate_request(
        self,
        *,
        request_id: str,
        messages: list[dict],
        approx_request_tokens: int,
        current_model_calls: int,
        logical_call_id: str | None = None,
        attempt_id: str | None = None,
        request_kind: str = "primary",
    ) -> dict[str, Any]:
        context_tokens = estimate_messages_tokens(messages)
        self.record_context_manifest(messages)
        if self.state is not None:
            self.state.usage.context_tokens_max = max(self.state.usage.context_tokens_max, context_tokens)
        action = self._threshold_action(
            self._projected_model_ratio(approx_request_tokens, current_model_calls, context_tokens)
        )
        if self._is_trusted_qa_actor() and action in {"informational", "warning", "approval"}:
            action = "allow"
        if self._qa_headroom_violation(projected_total_tokens=approx_request_tokens, projected_model_calls=1):
            action = "approval" if action != "hard_stop" else action
        payload = {
            "request_id": request_id,
            "logical_call_id": logical_call_id,
            "attempt_id": attempt_id,
            "request_kind": request_kind,
            "action": action,
            "context_tokens": context_tokens,
            "approx_request_tokens": approx_request_tokens,
            "api_call_count": current_model_calls,
        }
        pre = self._request_snapshot_view(self._root_snapshot(), request_id) if self._budget_file is not None else {}
        if self._budget_file is not None:
            def mutate(data: dict) -> dict:
                entry = self._request_state_entry(data, request_id)
                logical_entry = self._logical_call_entry(data, logical_call_id) if logical_call_id else None
                self._assert_request_transition(str(entry.get("state") or "created"), "evaluated", request_id=request_id)
                entry.update(
                    {
                        "state": "evaluated",
                        "decision": action,
                        "logical_call_id": logical_call_id,
                        "attempt_id": attempt_id,
                        "request_kind": request_kind,
                        "reservation_estimate": int(approx_request_tokens),
                        "updated_at": _utc_now(),
                    }
                )
                if logical_entry is not None:
                    request_ids = list(logical_entry.get("request_ids") or [])
                    if request_id not in request_ids:
                        request_ids.append(request_id)
                    logical_entry.update(
                        {
                            "request_ids": request_ids,
                            "last_request_id": request_id,
                            "last_attempt_id": attempt_id,
                            "preflight_count": int(logical_entry.get("preflight_count", 0)) + 1,
                            "last_context_tokens": int(context_tokens),
                            "last_context_fingerprint": self._context_fingerprint(messages),
                            "updated_at": _utc_now(),
                        }
                    )
                    if (
                        request_kind == "primary"
                        and action == "warning"
                        and int(logical_entry.get("preflight_count", 0))
                        >= int(self.config.progress.logical_call_max_preflights)
                    ):
                        entry["decision"] = "hard_stop"
                        logical_entry["termination_reason"] = "logical_call_preflight_limit"
                        payload["action"] = "hard_stop"
                return self._refresh_budget_pools(data)
            self._budget_file.update(mutate)
        action = payload.get("action", action)
        authorization = None
        if self.should_enforce() and action in {"approval", "hard_stop"}:
            authorization = self._consume_budget_authorization(
                request_id=request_id,
                logical_call_id=logical_call_id,
                approx_request_tokens=approx_request_tokens,
            )
        if self.mode == "observe":
            contract = self._decision_payload(
                request_id=request_id,
                action=action,
                allowed=True,
                reason=f"threshold_{action}_observed",
            )
        elif authorization is not None:
            contract = self._decision_payload(
                request_id=request_id,
                action=action,
                allowed=True,
                reason="human_budget_authorization",
                authorization_id=authorization.get("authorization_id"),
            )
        elif action == "warning" and request_kind == "primary":
            contract = self._decision_payload(
                request_id=request_id,
                action=action,
                allowed=False,
                reason="threshold_warning_requires_compaction",
                compact_context=True,
                termination_reason="threshold_warning",
            )
        elif action in {"approval", "hard_stop"}:
            contract = self._decision_payload(
                request_id=request_id,
                action=action,
                allowed=False,
                reason=f"threshold_{action}_requires_human_authorization",
                termination_reason="budget_approval_required" if action == "approval" else "cost_context_governance_hard_stop",
            )
        else:
            contract = self._decision_payload(
                request_id=request_id,
                action=action,
                allowed=True,
                reason=f"threshold_{action}",
            )
        payload.update(contract)
        if action != "allow":
            self.log_event("threshold_event", payload)
        after = self._request_snapshot_view(self._root_snapshot(), request_id) if self._budget_file is not None else {}
        self._log_request_transition({
            **payload,
            "transition": "evaluated",
            "phase": "preflight",
            "decision": action,
            "state_before": pre.get("request_state"),
            "state_after": after.get("request_state"),
            "consumed_before": {"total_tokens": pre.get("consumed_total_tokens", 0), "model_calls": pre.get("consumed_model_calls", 0)},
            "consumed_after": {"total_tokens": after.get("consumed_total_tokens", 0), "model_calls": after.get("consumed_model_calls", 0)},
            "reservation_estimate": int(approx_request_tokens),
            "outstanding_reservations": after.get("open_reservations", []),
        })
        return payload

    def reserve_request(self, request_id: str, estimate: int) -> dict[str, Any]:
        if self._budget_file is None:
            return {"allowed": True, "request_id": request_id, "reservation_id": request_id, "existing": False}

        def mutate(data: dict) -> dict:
            reservations = data.setdefault("reservations", {})
            entry = self._request_state_entry(data, request_id)
            current_state = str(entry.get("state") or "created")
            if current_state in self._request_terminal_states() or current_state in {"dispatched", "reconciled"}:
                raise RuntimeError(f"Cannot reserve request from state {current_state}: {request_id}")
            if request_id in reservations:
                existing = reservations[request_id]
                self._assert_request_transition(current_state, "reserved", request_id=request_id)
                entry.update(
                    {
                        "state": "reserved",
                        "reservation_id": request_id,
                        "reservation_estimate": int(existing.get("total_tokens", estimate)),
                        "updated_at": _utc_now(),
                    }
                )
                data["last_reservation"] = {
                    "request_id": request_id,
                    "reservation_id": request_id,
                    "existing": True,
                    "allowed": True,
                }
                return self._refresh_budget_pools(data)
            consumed = data.setdefault("consumed", asdict(EnvelopeUsage()))
            budget_data = data.get("budget", {}) or asdict(self._profile_budget())
            reserve = data.get("qa_reserve", {}) or {}
            reserved_tokens = sum(int((item or {}).get("total_tokens", 0)) for item in reservations.values())
            reserved_calls = sum(int((item or {}).get("model_calls", 0)) for item in reservations.values())
            current_tokens = int(consumed.get("total_tokens", 0)) + reserved_tokens
            current_calls = int(consumed.get("model_calls", 0)) + reserved_calls
            total_tokens = int(budget_data.get("total_tokens", self._profile_budget().total_tokens))
            total_calls = int(budget_data.get("total_model_calls", self._profile_budget().total_model_calls))
            authorization = entry.get("budget_authorization") if isinstance(entry.get("budget_authorization"), dict) else None
            authorized_envelope = (
                self._validate_authorization_envelope(authorization.get("envelope"))
                if authorization is not None
                else None
            )
            if self.mode == "observe":
                allowed = True
            elif authorized_envelope is not None:
                if authorized_envelope["operation"] == "increment":
                    token_cap = total_tokens + int(authorized_envelope["total_tokens"])
                    call_cap = total_calls + int(authorized_envelope["total_model_calls"])
                else:
                    token_cap = int(authorized_envelope["total_tokens"])
                    call_cap = int(authorized_envelope["total_model_calls"])
                allowed = current_tokens + int(estimate) <= token_cap and current_calls + 1 <= call_cap
            elif self._is_trusted_qa_actor():
                allowed = current_tokens + int(estimate) <= total_tokens and current_calls + 1 <= total_calls
            else:
                token_cap = max(0, total_tokens - int(reserve.get("total_tokens", 0)))
                call_cap = max(0, total_calls - int(reserve.get("model_calls", 0)))
                allowed = current_tokens + int(estimate) <= token_cap and current_calls + 1 <= call_cap
            entry.update({"decision": entry.get("decision") or "allow"})
            if not allowed:
                self._assert_request_transition(current_state, "blocked", request_id=request_id)
                entry.update(
                    {
                        "state": "blocked",
                        "reservation_id": None,
                        "reservation_estimate": int(estimate),
                        "termination_reason": "reservation_denied",
                        "updated_at": _utc_now(),
                    }
                )
                data["last_reservation"] = {
                    "request_id": request_id,
                    "reservation_id": None,
                    "existing": False,
                    "allowed": False,
                }
                return self._refresh_budget_pools(data)
            reservations[request_id] = {
                "request_id": request_id,
                "task_id": self.task_id,
                "agent_role": self.role,
                "total_tokens": int(estimate),
                "model_calls": 1,
                "reserved_at": _utc_now(),
            }
            self._assert_request_transition(current_state, "reserved", request_id=request_id)
            entry.update(
                {
                    "state": "reserved",
                    "reservation_id": request_id,
                    "reservation_estimate": int(estimate),
                    "updated_at": _utc_now(),
                }
            )
            data["last_reservation"] = {
                "request_id": request_id,
                "reservation_id": request_id,
                "existing": False,
                "allowed": True,
            }
            return self._refresh_budget_pools(data)

        pre = self._request_snapshot_view(self._root_snapshot(), request_id) if self._budget_file is not None else {}
        updated = self._budget_file.update(mutate)
        details = (updated or {}).get("last_reservation", {})
        allowed = bool(details.get("allowed", False))
        after = self._request_snapshot_view(updated or {}, request_id)
        self._log_request_transition({
            "request_id": request_id,
            "transition": "reserved" if allowed else "blocked",
            "phase": "reserve",
            "state_before": pre.get("request_state"),
            "state_after": after.get("request_state"),
            "decision": "allow" if allowed else "block",
            "reservation_id": details.get("reservation_id"),
            "reservation_estimate": int(after.get("reservation_estimate", estimate) or estimate),
            "reservation_existing": bool(details.get("existing", False)),
            "reservation_allowed": allowed,
            "consumed_before": {"total_tokens": pre.get("consumed_total_tokens", 0), "model_calls": pre.get("consumed_model_calls", 0)},
            "consumed_after": {"total_tokens": after.get("consumed_total_tokens", 0), "model_calls": after.get("consumed_model_calls", 0)},
            "budget_after": (updated or {}).get("pools", {}),
            "outstanding_reservations": after.get("open_reservations", []),
        })
        return {
            "allowed": allowed,
            "request_id": request_id,
            "reservation_id": details.get("reservation_id"),
            "existing": bool(details.get("existing", False)),
            "budget_after": (updated or {}).get("pools", {}),
        }

    def record_dispatch(self, request_id: str) -> dict[str, Any]:
        if self._budget_file is None:
            return {}

        def mutate(data: dict) -> dict:
            reservations = data.setdefault("reservations", {})
            if request_id not in reservations:
                raise RuntimeError(f"Cannot dispatch request without reservation: {request_id}")
            entry = self._request_state_entry(data, request_id)
            self._assert_request_transition(str(entry.get("state") or "created"), "dispatched", request_id=request_id)
            entry["state"] = "dispatched"
            entry["dispatch_count"] = int(entry.get("dispatch_count", 0)) + 1
            entry["updated_at"] = _utc_now()
            return self._refresh_budget_pools(data)

        pre = self._request_snapshot_view(self._root_snapshot(), request_id) if self._budget_file is not None else {}
        updated = self._budget_file.update(mutate)
        after = self._request_snapshot_view(updated or {}, request_id)
        self._log_request_transition({
            "request_id": request_id,
            "transition": "dispatched",
            "phase": "dispatch",
            "state_before": pre.get("request_state"),
            "state_after": after.get("request_state"),
            "consumed_before": {"total_tokens": pre.get("consumed_total_tokens", 0), "model_calls": pre.get("consumed_model_calls", 0)},
            "consumed_after": {"total_tokens": after.get("consumed_total_tokens", 0), "model_calls": after.get("consumed_model_calls", 0)},
            "reservation_id": request_id,
            "reservation_estimate": after.get("reservation_estimate"),
            "outstanding_reservations": after.get("open_reservations", []),
        })
        return updated

    def handle_dispatch_persistence_failure(self, request_id: str, exc: BaseException) -> None:
        """Account a failed dispatch transition and fail closed on every error."""
        self._dispatch_failure_latched = True
        metadata = _exception_metadata(exc)
        try:
            self.account_dispatched_request_failure(
                request_id, reason="accounting_error", error_message=exc
            )
        except Exception as accounting_exc:
            metadata = _exception_metadata(accounting_exc)
        for operation in (
            lambda: self.persist_checkpoint("dispatch-persistence-error", {
                "request_id": request_id,
                "reason": "dispatch_persistence_error",
                "error": metadata,
            }),
            lambda: self.persist_partial_handoff(
                task=self.task_id,
                status="accounting_error",
                summary="Governança interrompeu a chamada antes do provider por falha contábil.",
                limitations=["A confirmação persistida do dispatch não foi obtida."],
                errors=[metadata],
                recommended_next_step="Reconciliar o estado contábil antes de retomar.",
            ),
        ):
            try:
                operation()
            except Exception:
                self._dispatch_failure_latched = True
        raise RuntimeError("Governança não confirmou o dispatch persistido; provider bloqueado.") from exc

    def release_request(self, request_id: str, reason: str) -> dict[str, Any]:
        if self._budget_file is None:
            return {}

        terminal_state = "released_before_dispatch"

        def mutate(data: dict) -> dict:
            reservations = data.setdefault("reservations", {})
            entry = self._request_state_entry(data, request_id)
            self._assert_request_transition(str(entry.get("state") or "created"), terminal_state, request_id=request_id)
            reservations.pop(request_id, None)
            entry.update(
                {
                    "state": terminal_state,
                    "released_reason": reason,
                    "termination_reason": reason,
                    "updated_at": _utc_now(),
                }
            )
            return self._refresh_budget_pools(data)

        pre = self._request_snapshot_view(self._root_snapshot(), request_id) if self._budget_file is not None else {}
        updated = self._budget_file.update(mutate)
        after = self._request_snapshot_view(updated or {}, request_id)
        self._log_request_transition({
            "request_id": request_id,
            "transition": terminal_state,
            "phase": "finalize",
            "state_before": pre.get("request_state"),
            "state_after": after.get("request_state"),
            "termination_reason": reason,
            "consumed_before": {"total_tokens": pre.get("consumed_total_tokens", 0), "model_calls": pre.get("consumed_model_calls", 0)},
            "consumed_after": {"total_tokens": after.get("consumed_total_tokens", 0), "model_calls": after.get("consumed_model_calls", 0)},
            "outstanding_reservations": after.get("open_reservations", []),
        })
        return updated

    def _canonical_from_reservation(
        self,
        reservation_estimate: int,
        *,
        response_text: str = "",
    ) -> dict[str, Any]:
        return self.normalize_usage_payload(
            None,
            approx_request_tokens=max(1, int(reservation_estimate or 0)),
            response_text=response_text,
        )

    def _terminal_accounting_state(self, reason: str) -> str:
        mapping = {
            "provider_error": "provider_error_accounted",
            "interrupted": "interrupted_accounted",
            "interrupted_redirect": "interrupted_accounted",
            "accounting_error": "accounting_error",
        }
        return mapping.get(reason, reason)

    def account_dispatched_request_failure(
        self,
        request_id: str,
        *,
        reason: str,
        usage: Any | None = None,
        duration_seconds: float = 0.0,
        retry_count: int = 0,
        response_text: str = "",
        error_message: Any = None,
    ) -> dict[str, Any]:
        if self._budget_file is None:
            return {}

        terminal_state = self._terminal_accounting_state(reason)
        now = _utc_now()

        def mutate(data: dict) -> dict:
            reservations = data.setdefault("reservations", {})
            request_state = data.setdefault("request_state", {})
            entry = request_state.setdefault(request_id, self._request_state_entry(data, request_id))
            current_state = str(entry.get("state") or "")
            if current_state == terminal_state:
                return self._refresh_budget_pools(data)
            self._assert_request_transition(current_state, terminal_state, request_id=request_id)
            reservation = reservations.pop(request_id, None) or {}
            reservation_estimate = int(
                reservation.get("total_tokens", 0)
                or entry.get("reservation_estimate", 0)
                or 0
            )
            canonical = self.normalize_usage_payload(
                usage,
                approx_request_tokens=max(1, reservation_estimate),
                response_text=response_text,
            )
            consumed = data.setdefault("consumed", asdict(EnvelopeUsage()))
            children = data.setdefault("children", {})
            child = children.setdefault(self.task_id, asdict(EnvelopeUsage()))
            deltas = {
                "model_calls": max(1, int(reservation.get("model_calls", 1) or 1)),
                "retries": int(retry_count),
                "input_tokens": canonical["input_tokens"],
                "output_tokens": canonical["output_tokens"],
                "total_tokens": canonical["total_tokens"],
            }
            for key, value in deltas.items():
                consumed[key] = int(consumed.get(key, 0)) + int(value)
                child[key] = int(child.get(key, 0)) + int(value)
            child["updated_at"] = now
            entry.update(
                {
                    "state": terminal_state,
                    "termination_reason": reason,
                    "reservation_id": request_id,
                    "reservation_estimate": reservation_estimate,
                    "reconciliation": {
                        "reserved_tokens": reservation_estimate,
                        "reserved_calls": int(reservation.get("model_calls", 1) or 1),
                        "actual_total_tokens": canonical["total_tokens"],
                        "actual_model_calls": 1,
                        "estimated": bool(canonical.get("estimated", False)),
                    },
                    "canonical_usage": dict(canonical),
                    "accounting_error": _exception_metadata(error_message),
                    "updated_at": now,
                }
            )
            data["reservation_reconciliation"] = {
                "request_id": request_id,
                "task_id": self.task_id,
                "reserved_tokens": reservation_estimate,
                "reserved_calls": int(reservation.get("model_calls", 1) or 1),
                "actual_total_tokens": canonical["total_tokens"],
                "actual_model_calls": 1,
                "estimated": bool(canonical.get("estimated", False)),
                "terminal_state": terminal_state,
            }
            if terminal_state == "accounting_error":
                self._activate_accounting_latch(data, request_id=request_id, reason="accounting_error")
            return self._refresh_budget_pools(data)

        pre = self._request_snapshot_view(self._root_snapshot(), request_id) if self._budget_file is not None else {}
        updated = self._budget_file.update(mutate)
        self.persist_checkpoint(
            f"{terminal_state}-post-dispatch",
            {
                "request_id": request_id,
                "reason": reason,
                "error": _exception_metadata(error_message),
                "duration_seconds": duration_seconds,
                "retry_count": retry_count,
            },
        )
        after = self._request_snapshot_view(updated or {}, request_id)
        self._log_request_transition(
            {
                "request_id": request_id,
                "transition": terminal_state,
                "phase": "finalize",
                "state_before": pre.get("request_state"),
                "state_after": after.get("request_state"),
                "termination_reason": reason,
                "error": _exception_metadata(error_message),
                "reservation_id": request_id,
                "reservation_estimate": after.get("reservation_estimate"),
                "consumed_before": {"total_tokens": pre.get("consumed_total_tokens", 0), "model_calls": pre.get("consumed_model_calls", 0)},
                "consumed_after": {"total_tokens": after.get("consumed_total_tokens", 0), "model_calls": after.get("consumed_model_calls", 0)},
                "outstanding_reservations": after.get("open_reservations", []),
            }
        )
        return updated

    def reconcile_usage(
        self,
        request_id: str,
        usage: Any | None,
        *,
        duration_seconds: float,
        retry_count: int = 0,
        approx_request_tokens: int = 0,
        response_text: str = "",
    ) -> dict[str, Any]:
        snapshot = self._root_snapshot() if self._budget_file is not None else {}
        entry = ((snapshot.get("request_state") or {}).get(request_id) or {})
        if str(entry.get("state") or "") in {"reconciled", "completed"} and isinstance(entry.get("canonical_usage"), dict):
            return dict(entry.get("canonical_usage") or {})
        reservation = ((snapshot.get("reservations") or {}).get(request_id) or {})
        persisted_estimate = int(
            reservation.get("total_tokens", 0)
            or entry.get("reservation_estimate", 0)
            or approx_request_tokens
            or 0
        )
        canonical = self.normalize_usage_payload(
            usage,
            approx_request_tokens=max(1, persisted_estimate),
            response_text=response_text,
        )
        if self._budget_file is not None:
            pre = self._request_snapshot_view(snapshot, request_id)

            def reconcile(data: dict) -> dict:
                reservations = data.setdefault("reservations", {})
                request_state = data.setdefault("request_state", {})
                entry = request_state.setdefault(request_id, self._request_state_entry(data, request_id))
                current_state = str(entry.get("state") or "")
                if current_state in {"reconciled", "completed"} and entry.get("reconciliation"):
                    data["reservation_reconciliation"] = {
                        "request_id": request_id,
                        "task_id": self.task_id,
                        **dict(entry.get("reconciliation") or {}),
                    }
                    return self._refresh_budget_pools(data)
                self._assert_request_transition(current_state, "reconciled", request_id=request_id)
                reservation = reservations.pop(request_id, None) or {}
                reserved_tokens = int(reservation.get("total_tokens", 0) or entry.get("reservation_estimate", 0) or persisted_estimate)
                reserved_calls = int(reservation.get("model_calls", 0) or 1)
                consumed = data.setdefault("consumed", asdict(EnvelopeUsage()))
                children = data.setdefault("children", {})
                child = children.setdefault(self.task_id, asdict(EnvelopeUsage()))
                deltas = {
                    "model_calls": 1,
                    "retries": int(retry_count),
                    "input_tokens": canonical["input_tokens"],
                    "output_tokens": canonical["output_tokens"],
                    "total_tokens": canonical["total_tokens"],
                }
                for key, value in deltas.items():
                    consumed[key] = int(consumed.get(key, 0)) + int(value)
                    child[key] = int(child.get(key, 0)) + int(value)
                child["updated_at"] = _utc_now()
                entry.update(
                    {
                        "state": "reconciled",
                        "reconciliation": {
                            "reserved_tokens": reserved_tokens,
                            "reserved_calls": reserved_calls,
                            "actual_total_tokens": canonical["total_tokens"],
                            "actual_model_calls": 1,
                            "estimated": bool(canonical.get("estimated", False)),
                        },
                        "canonical_usage": dict(canonical),
                        "updated_at": _utc_now(),
                    }
                )
                data["reservation_reconciliation"] = {
                    "request_id": request_id,
                    "task_id": self.task_id,
                    "reserved_tokens": reserved_tokens,
                    "reserved_calls": reserved_calls,
                    "actual_total_tokens": canonical["total_tokens"],
                    "actual_model_calls": 1,
                    "estimated": bool(canonical.get("estimated", False)),
                }
                return self._refresh_budget_pools(data)

            updated = self._budget_file.update(reconcile)
            after = self._request_snapshot_view(updated or {}, request_id)
        else:
            updated = {}
            pre = {}
            after = {}

        self._log_request_transition({
            "request_id": request_id,
            "transition": "reconciled",
            "phase": "reconcile",
            "state_before": pre.get("request_state"),
            "state_after": after.get("request_state"),
            "actual_total_tokens": canonical["total_tokens"],
            "reservation_estimate": int(persisted_estimate),
            "consumed_before": {"total_tokens": pre.get("consumed_total_tokens", 0), "model_calls": pre.get("consumed_model_calls", 0)},
            "consumed_after": {"total_tokens": after.get("consumed_total_tokens", 0), "model_calls": after.get("consumed_model_calls", 0)},
            "outstanding_reservations": after.get("open_reservations", []),
        })

        if self.state is not None:
            for key, value in {
                "model_calls": 1,
                "retries": int(retry_count),
                "input_tokens": canonical["input_tokens"],
                "output_tokens": canonical["output_tokens"],
                "total_tokens": canonical["total_tokens"],
            }.items():
                if hasattr(self.state.usage, key):
                    setattr(self.state.usage, key, int(getattr(self.state.usage, key)) + int(value))
            self.state.updated_at = _utc_now()
        if response_text and response_text.strip():
            self._consume(progress_events=1)
        else:
            self._consume(no_progress_iterations=1)
        self.log_event("model_call", {
            "provider": getattr(self.agent, "provider", None),
            "model": getattr(self.agent, "model", None),
            "request_id": request_id,
            "request_class": self.request_class,
            "duration_seconds": duration_seconds,
            "input_tokens": canonical["input_tokens"],
            "output_tokens": canonical["output_tokens"],
            "total_tokens": canonical["total_tokens"],
            "cached_input_tokens": canonical["cached_input_tokens"] if canonical["cache_confirmed"] else 0,
            "cache_creation_tokens": canonical["cache_creation_tokens"] if canonical["cache_confirmed"] else 0,
            "reasoning_tokens": canonical["reasoning_tokens"],
            "estimated_tokens": canonical["estimated"],
            "cache_confirmed": canonical["cache_confirmed"],
            "context_manifest": self._context_manifest,
            "retry_count": retry_count,
        })
        return canonical

    def finalize_request(self, request_id: str, outcome: str, termination_reason: str | None = None) -> dict[str, Any]:
        if self._budget_file is None:
            return {}

        outcome_to_state = {
            "success": "completed",
            "blocked": "blocked",
            "released": "released_before_dispatch",
            "interrupted": "interrupted_accounted",
            "provider_error": "provider_error_accounted",
            "accounting_error": "accounting_error",
            "completed": "completed",
            "released_before_dispatch": "released_before_dispatch",
            "interrupted_accounted": "interrupted_accounted",
            "provider_error_accounted": "provider_error_accounted",
            "blocked": "blocked",
        }
        next_state = outcome_to_state.get(outcome, outcome or "accounting_error")

        def mutate(data: dict) -> dict:
            entry = self._request_state_entry(data, request_id)
            current_state = str(entry.get("state") or "created")
            if current_state == next_state:
                entry["termination_reason"] = termination_reason or entry.get("termination_reason")
                entry["updated_at"] = _utc_now()
                return self._refresh_budget_pools(data)
            self._assert_request_transition(current_state, next_state, request_id=request_id)
            entry["state"] = next_state
            entry["termination_reason"] = termination_reason or entry.get("termination_reason")
            entry["updated_at"] = _utc_now()
            return self._refresh_budget_pools(data)

        updated = self._budget_file.update(mutate)
        self._log_request_transition({
            "request_id": request_id,
            "transition": next_state,
            "termination_reason": termination_reason,
            "outstanding_reservations": list(((updated or {}).get("reservations") or {}).keys()),
        })
        return updated

    def _qa_headroom_violation(self, projected_total_tokens: int = 0, projected_model_calls: int = 0) -> bool:
        root = self._root_snapshot()
        budget_data = root.get("budget", {}) or asdict(self._profile_budget())
        reserve = root.get("qa_reserve", {})
        consumed = root.get("consumed", {})
        reserved = root.get("reservations", {})
        reserved_tokens = sum(int((item or {}).get("total_tokens", 0)) for item in reserved.values())
        reserved_calls = sum(int((item or {}).get("model_calls", 0)) for item in reserved.values())
        total_tokens = int(budget_data.get("total_tokens", self._profile_budget().total_tokens))
        total_calls = int(budget_data.get("total_model_calls", self._profile_budget().total_model_calls))
        current_tokens = int(consumed.get("total_tokens", 0)) + reserved_tokens
        current_calls = int(consumed.get("model_calls", 0)) + reserved_calls
        if self._is_trusted_qa_actor():
            return current_tokens + int(projected_total_tokens) > total_tokens or current_calls + int(projected_model_calls) > total_calls
        non_qa_token_cap = max(0, total_tokens - int(reserve.get("total_tokens", 0)))
        non_qa_call_cap = max(0, total_calls - int(reserve.get("model_calls", 0)))
        return current_tokens + int(projected_total_tokens) > non_qa_token_cap or current_calls + int(projected_model_calls) > non_qa_call_cap

    def _threshold_action(self, ratio: float) -> str:
        t = self.config.threshold
        if ratio >= t.hard_stop_ratio:
            return "hard_stop"
        if ratio >= t.approval_ratio:
            return "approval"
        if ratio >= t.warning_ratio:
            return "warning"
        if ratio >= t.informational_ratio:
            return "informational"
        return "allow"

    def _projected_model_ratio(self, approx_request_tokens: int, api_call_count: int, context_tokens: int) -> float:
        budget = self._profile_budget()
        root = self._root_consumed()
        return max(
            context_tokens / max(1, int(budget.context_tokens_per_request)),
            approx_request_tokens / max(1, int(budget.context_tokens_per_request)),
            (api_call_count + 1) / max(1, int(budget.calls_per_agent)),
            (int(root.get("total_tokens", 0)) + int(approx_request_tokens)) / max(1, int(budget.total_tokens)),
            (int(root.get("model_calls", 0)) + 1) / max(1, int(budget.total_model_calls)),
            (time.time() - self._turn_started_at) / max(1, int(budget.wall_clock_seconds)),
        )

    def before_model_call(
        self,
        *,
        request_id: str,
        messages: list[dict],
        approx_request_tokens: int,
        api_call_count: int,
        logical_call_id: str | None = None,
        attempt_id: str | None = None,
        request_kind: str = "primary",
    ) -> dict[str, Any]:
        if not self.is_active():
            return self._decision_payload(
                request_id=request_id,
                action="disabled",
                allowed=True,
                reason="operator_disabled",
            )
        if self._artifact_failure is not None and self.should_enforce():
            return self._decision_payload(
                request_id=request_id,
                action="hard_stop",
                allowed=False,
                reason="invalid_governance_artifact",
                termination_reason="invalid_governance_artifact",
                stop=True,
                message="Governança bloqueou a execução porque um artifact decisório está inválido.",
            )
        if self.state is None or self._budget_file is None:
            inferred_objective = ""
            for msg in reversed(messages or []):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    inferred_objective = str(msg.get("content") or "")
                    break
            seed = getattr(self.agent, "_governance_seed", None)
            self.identity = self.resolve_or_create_engagement_identity()
            self.root_dir = self.identity.persistence_root
            self.engagement_id = self.identity.engagement_id
            budget = self._profile_budget()
            if isinstance(seed, dict) and isinstance(seed.get("budget_override"), dict):
                budget = BudgetProfile.from_mapping(seed.get("budget_override"), fallback=budget)
            qa_tokens = int(budget.total_tokens * float(budget.qa_reserve_ratio or 0.0))
            qa_calls = int(budget.total_model_calls * float(budget.qa_reserve_ratio or 0.0))
            self.state = EnvelopeState(
                engagement_id=self.engagement_id,
                task_id=self.task_id,
                request_class=self.request_class,
                profile_name=self.profile_key,
                role=self.role,
                status="active",
                objective=inferred_objective[:500],
                selected_agents=list(seed.get("selected_agents") or []) if isinstance(seed, dict) else [],
                qa_reserve={"total_tokens": qa_tokens, "model_calls": qa_calls},
                budget=asdict(budget),
            )
            self._budget_file = FileBackedBudget(self._engagement_dir() / "budget.json")
            if not self._initialized:
                try:
                    self._budget_file.initialize(self._bootstrap_budget({}, budget))
                except GovernanceArtifactError as exc:
                    self._record_artifact_failure(exc, phase="lazy_initialization")
                    if not self.should_enforce():
                        return self._decision_payload(
                            request_id=request_id,
                            action="informational",
                            allowed=True,
                            reason="invalid_governance_artifact_observed",
                        )
                    return self._decision_payload(
                        request_id=request_id,
                        action="hard_stop",
                        allowed=False,
                        reason="invalid_governance_artifact",
                        termination_reason="invalid_governance_artifact",
                        stop=True,
                        message="Governança bloqueou a execução porque um artifact decisório está inválido.",
                    )
                self._initialized = True
            self._persist_json("brief.json", {
                "engagement_id": self.engagement_id,
                "request_class": self.request_class,
                "objective": _text_metadata(inferred_objective),
                "scope": [],
                "scope_exclusions": [],
                "decisions": [],
                "constraints": [],
                "selected_agents": list(seed.get("selected_agents") or []) if isinstance(seed, dict) else [],
                "resource_envelope": asdict(budget),
                "qa_reserve": self.state.qa_reserve,
                "qa_obligations": [],
                "status": self.state.status,
                "artifact_references": [],
                "evidence_ids": [],
                "claim_ids": [],
                "evidence_ledger": "evidence_ledger.jsonl",
                "claim_ledger": "claim_ledger.jsonl",
                "open_questions": [],
                "checkpoints": [],
                "usage_summary": asdict(self.state.usage),
            })
            self.record_context_manifest(messages, system_message="")
            self.log_event("turn_begin", {
                "request_class": self.request_class,
                "profile": self.profile_key,
                "session_id": getattr(self.agent, "session_id", None),
                "agent_role": self.role,
                "selected_agents": self.state.selected_agents,
                "mode": self.mode,
                "lazy_initialized": True,
            })
            self._write_mode_status(active=True, reason="governance_active")
        resume_logical_call_id = getattr(self.agent, "_governance_resume_logical_call_id", None)
        if (
            isinstance(resume_logical_call_id, str)
            and resume_logical_call_id
            and isinstance(getattr(self.agent, "_governance_resume_authorization_id", None), str)
            and (logical_call_id is None or logical_call_id == resume_logical_call_id)
        ):
            logical_call_id = resume_logical_call_id
        try:
            payload = self.evaluate_request(
                request_id=request_id,
                messages=messages,
                approx_request_tokens=approx_request_tokens,
                current_model_calls=api_call_count,
                logical_call_id=logical_call_id,
                attempt_id=attempt_id,
                request_kind=request_kind,
            )
        except GovernanceArtifactError as exc:
            self._record_artifact_failure(exc, phase="before_model_call")
            if not self.should_enforce():
                return self._decision_payload(
                    request_id=request_id,
                    action="informational",
                    allowed=True,
                    reason="invalid_governance_artifact_observed",
                )
            return self._decision_payload(
                request_id=request_id,
                action="hard_stop",
                allowed=False,
                reason="invalid_governance_artifact",
                termination_reason="invalid_governance_artifact",
                stop=True,
                message="Governança bloqueou a execução porque um artifact decisório está inválido.",
            )
        payload = self.validate_decision(payload, request_id=request_id)
        accounting_latch = self._engagement_accounting_latch()
        if (self._dispatch_failure_latched or accounting_latch.get("active")) and self.should_enforce():
            self.persist_checkpoint(
                "hard-stop-accounting-error",
                {
                    "request_id": request_id,
                    "blocking_request_ids": accounting_latch.get("request_ids", []),
                    "accounting_latch": accounting_latch,
                },
            )
            handoff = self.persist_partial_handoff(
                task=self.task_id,
                status="hard_stop",
                summary="Hard stop: erro contábil pendente bloqueou nova chamada paga ao provider.",
                limitations=["An unresolved engagement accounting latch is active after a post-dispatch accounting failure."],
                recommended_next_step="Recover the accounting state or resume from the saved handoff before authorizing another paid provider call.",
                artifact_references=[],
            )
            self.finalize_request(
                request_id,
                "blocked",
                termination_reason="unresolved_accounting_latch",
            )
            return {
                **payload,
                "allowed": False,
                "action": "hard_stop",
                "reason": "unresolved_accounting_latch",
                "compact_context": False,
                "termination_reason": "unresolved_accounting_latch",
                "stop": True,
                "message": "Governança bloqueou novas chamadas até reconciliar erro contábil anterior.",
                "resume_identifier": handoff.get("resume_identifier"),
                "artifact_references": handoff.get("artifact_references", []),
            }
        action = payload["action"]
        if not payload["allowed"] and payload["compact_context"]:
            return payload
        if not payload["allowed"] and action in {"approval", "hard_stop"}:
            checkpoint_label = "approval-before-model" if action == "approval" else "hard-stop-before-model"
            self.persist_checkpoint(checkpoint_label, {
                "messages_tail": messages[-6:],
                "approx_request_tokens": approx_request_tokens,
                "context_manifest": self._context_manifest,
            })
            proposed_envelope = {
                "operation": "increment",
                "total_tokens": max(1, int(approx_request_tokens)),
                "total_model_calls": 1,
                "context_tokens_per_request": max(1, int(approx_request_tokens)),
            }
            authorization = self.create_budget_authorization(
                blocked_request_id=request_id,
                logical_call_id=logical_call_id,
                decision_action=action,
                proposed_envelope=proposed_envelope,
            )
            summary = (
                "Governança exigiu autorização humana antes de continuar."
                if action == "approval"
                else "Hard stop de custo/contexto atingido antes da próxima chamada ao modelo."
            )
            handoff = self.persist_partial_handoff(
                task=self.task_id,
                status="approval" if action == "approval" else "hard_stop",
                summary=summary,
                limitations=["Automation is suspended until an authenticated owner callback authorizes an explicit request-scoped envelope."],
                recommended_next_step=authorization["instructions"],
            )
            if action == "hard_stop":
                self.close_turn(
                    status="partial",
                    final_response=summary,
                    termination_reason="cost_context_governance_hard_stop",
                )
            message = f"{summary} {authorization['instructions']}"
            return {
                **payload,
                "pause": action == "approval",
                "stop": action == "hard_stop",
                "message": message,
                "authorization": authorization,
                "resume_identifier": handoff.get("resume_identifier"),
                "artifact_references": handoff.get("artifact_references", []),
            }
        if not payload["allowed"]:
            return {**payload, "stop": True, "message": payload["termination_reason"] or payload["reason"]}

        reservation = self.reserve_request(request_id, approx_request_tokens)
        if not reservation.get("allowed"):
                self.persist_checkpoint("hard-stop-before-model", {
                    "messages_tail": messages[-6:],
                    "approx_request_tokens": approx_request_tokens,
                    "context_manifest": self._context_manifest,
                    "denied_by_budget_gate": True,
                })
                handoff = self.persist_partial_handoff(
                    task=self.task_id,
                    status="hard_stop",
                    summary="Hard stop de custo/contexto atingido antes da próxima chamada ao modelo.",
                    limitations=["Atomic budget admission rejected the next model call before provider dispatch."],
                    recommended_next_step="Resume from the saved handoff after reducing context, narrowing scope, or explicitly increasing the envelope.",
                )
                self.finalize_request(request_id, "blocked", "hard_stop")
                self.close_turn(
                    status="partial",
                    final_response="Hard stop de custo/contexto atingido antes da próxima chamada ao modelo.",
                    termination_reason="cost_context_governance_hard_stop",
                )
                return {
                    **payload,
                    "allowed": False,
                    "action": "hard_stop",
                    "reason": "atomic_budget_admission_denied",
                    "compact_context": False,
                    "termination_reason": "cost_context_governance_hard_stop",
                    "stop": True,
                    "message": "Hard stop de custo/contexto atingido antes da próxima chamada ao modelo.",
                    "resume_identifier": handoff.get("resume_identifier"),
                    "artifact_references": handoff.get("artifact_references", []),
                    "reservation_id": reservation.get("reservation_id"),
                }
        return self.validate_decision(
            {**payload, "reservation_id": reservation.get("reservation_id")},
            request_id=request_id,
        )

    def normalize_usage_payload(self, usage: Any | None, *, approx_request_tokens: int = 0, response_text: str = "") -> dict[str, Any]:
        payload = _coerce_dict(usage)
        candidates = payload or usage or {}
        canonical = {
            "input_tokens": _safe_int(_lookup_nested(candidates, "input_tokens", "prompt_tokens", "input_tokens_total", "usage.input_tokens", "usage.prompt_tokens", "usage.input_tokens_total"), 0),
            "output_tokens": _safe_int(_lookup_nested(candidates, "output_tokens", "completion_tokens", "output_tokens_total", "usage.output_tokens", "usage.completion_tokens", "usage.output_tokens_total"), 0),
            "total_tokens": _safe_int(_lookup_nested(candidates, "total_tokens", "usage.total_tokens"), 0),
            "cached_input_tokens": _safe_int(_lookup_nested(candidates, "cached_input_tokens", "input_cached_tokens", "usage.cached_input_tokens", "prompt_tokens_details.cached_tokens"), 0),
            "cache_creation_tokens": _safe_int(_lookup_nested(candidates, "cache_creation_tokens", "cache_write_input_tokens", "usage.cache_creation_tokens"), 0),
            "reasoning_tokens": _safe_int(_lookup_nested(candidates, "reasoning_tokens", "output_tokens_details.reasoning_tokens", "usage.reasoning_tokens"), 0),
            "estimated": False,
            "cache_confirmed": False,
        }
        if canonical["total_tokens"] <= 0:
            canonical["total_tokens"] = canonical["input_tokens"] + canonical["output_tokens"]
        if canonical["cached_input_tokens"] > 0 or canonical["cache_creation_tokens"] > 0:
            canonical["cache_confirmed"] = True
        if canonical["total_tokens"] <= 0 or canonical["input_tokens"] < 0 or canonical["output_tokens"] < 0:
            canonical["estimated"] = True
            canonical["input_tokens"] = int(max(1, approx_request_tokens) * _USAGE_MARGIN_INPUT)
            canonical["output_tokens"] = max(32, int(estimate_text_tokens(response_text) * _USAGE_MARGIN_OUTPUT))
            canonical["total_tokens"] = canonical["input_tokens"] + canonical["output_tokens"]
            canonical["cached_input_tokens"] = 0
            canonical["cache_creation_tokens"] = 0
            canonical["reasoning_tokens"] = 0
            canonical["cache_confirmed"] = False
        return canonical

    def record_model_usage(self, usage: Any | None, *, request_id: str, duration_seconds: float, retry_count: int = 0, approx_request_tokens: int = 0, response_text: str = "") -> dict[str, Any]:
        if not self.is_active():
            return {}
        try:
            canonical = self.reconcile_usage(
                request_id,
                usage,
                duration_seconds=duration_seconds,
                retry_count=retry_count,
                approx_request_tokens=approx_request_tokens,
                response_text=response_text,
            )
            self.finalize_request(request_id, "success", "completed")
            return canonical
        except Exception as exc:
            self.account_dispatched_request_failure(
                request_id,
                reason="accounting_error",
                usage=usage,
                duration_seconds=duration_seconds,
                retry_count=retry_count,
                response_text=response_text,
                error_message=str(exc),
            )
            snapshot = self._root_snapshot()
            entry = ((snapshot.get("request_state") or {}).get(request_id) or {})
            reservation_estimate = int(entry.get("reservation_estimate", 0) or approx_request_tokens or 1)
            return self._canonical_from_reservation(reservation_estimate, response_text=response_text)

    def before_tool_call(self, tool_name: str, tool_args: dict) -> dict[str, Any]:
        signature = hash_context_block({"tool": tool_name, "args": tool_args})
        request_id = f"tool:{signature[:24]}"
        if not self.is_active():
            return self._decision_payload(
                request_id=request_id,
                action="disabled",
                allowed=True,
                reason="operator_disabled",
            )
        budget = self._profile_budget()
        root = self._root_consumed()
        projected = int(root.get("tool_invocations", 0)) + 1
        action = self._threshold_action(projected / max(1, int(budget.tool_invocations)))
        repeats = self._equivalent_tool_counts.get(signature, 0) + 1
        self._equivalent_tool_counts[signature] = repeats
        no_progress_limit = max(
            int(budget.iterations_without_progress),
            int(self.config.progress.no_progress_iterations),
        )
        if repeats >= int(self.config.progress.equivalent_tool_calls):
            action = "hard_stop" if self.should_enforce() else "warning"
        elif self.state is not None and self.state.usage.no_progress_iterations + 1 >= no_progress_limit:
            action = "hard_stop" if self.should_enforce() else "warning"
        if action in {"warning", "approval", "hard_stop"}:
            self.log_event("tool_threshold", {
                "tool_name": tool_name,
                "action": action,
                "tool_invocations_projected": projected,
                "equivalent_repeats": repeats,
            })
        if action == "hard_stop" and self.should_enforce():
            self.persist_checkpoint("hard-stop-before-tool", {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "equivalent_repeats": repeats,
            })
            return self._decision_payload(
                request_id=request_id,
                action="hard_stop",
                allowed=False,
                reason="tool_threshold_hard_stop",
                termination_reason="cost_context_governance_tool_hard_stop",
                message=f"Governança bloqueou nova chamada para {tool_name}.",
            )
        return self._decision_payload(
            request_id=request_id,
            action=action,
            allowed=True,
            reason=f"tool_threshold_{action}_observed" if self.mode == "observe" else f"tool_threshold_{action}",
        )

    def record_tool_result(self, tool_name: str, tool_args: dict, result: str, *, is_error: bool, duration_seconds: float) -> None:
        if not self.is_active():
            return
        self._tools_used.add(tool_name)
        no_progress_delta = 0
        progress_delta = 0
        error_signature = hash_context_block({
            "tool": tool_name,
            "result": result[:1000] if result else "",
            "error": is_error,
        })
        if is_error:
            repeats = self._error_counts.get(error_signature, 0) + 1
            self._error_counts[error_signature] = repeats
            no_progress_delta = 1
            if repeats >= int(self.config.progress.same_error_repeats):
                self.persist_checkpoint("repeated-tool-error", {
                    "tool_name": tool_name,
                    "repeats": repeats,
                    "result_preview": result[:500],
                })
        else:
            progress_delta = 1 if (result and str(result).strip()) else 0
            no_progress_delta = 0 if progress_delta else 1
            if progress_delta:
                self._error_counts.clear()
        self._consume(tool_invocations=1, no_progress_iterations=no_progress_delta, progress_events=progress_delta)
        self.log_event("tool_call", {
            "tool_name": tool_name,
            "duration_seconds": duration_seconds,
            "error": bool(is_error),
            "progress_delta": progress_delta,
            "no_progress_delta": no_progress_delta,
            "args_hash": hash_context_block(tool_args),
        })

    def allocate_child_budget(self, *, child_task_id: str, requested_profile: str | None = None, selected_agents: Optional[list[str]] = None, governance_role: str | None = None) -> dict[str, Any]:
        if not self.is_active():
            return {"profile": requested_profile or self.profile_key, "budget": asdict(self._profile_budget()), "allowed": True}
        root_profile = self.config.profiles.get(requested_profile or self.profile_key, self._profile_budget())
        child_budget = {
            "total_model_calls": max(1, min(int(root_profile.calls_per_delegated_task), int(root_profile.total_model_calls))),
            "calls_per_agent": max(1, int(root_profile.calls_per_delegated_task)),
            "calls_per_delegated_task": max(1, int(root_profile.calls_per_delegated_task)),
            "input_tokens": max(1000, int(root_profile.input_tokens * 0.4)),
            "output_tokens": max(500, int(root_profile.output_tokens * 0.4)),
            "total_tokens": max(2000, int(root_profile.total_tokens * 0.4)),
            "context_tokens_per_request": max(1000, int(root_profile.context_tokens_per_request * 0.7)),
            "wall_clock_seconds": max(60, int(root_profile.wall_clock_seconds * 0.7)),
            "tool_invocations": max(4, int(root_profile.tool_invocations * 0.4)),
            "retries": max(1, int(root_profile.retries)),
            "delegation_count": 0,
            "iterations_without_progress": max(2, int(root_profile.iterations_without_progress)),
            "qa_reserve_ratio": 0.0,
        }
        child_role = str(governance_role or "sme")
        allowed = not self._qa_headroom_violation(
            projected_total_tokens=child_budget["total_tokens"],
            projected_model_calls=child_budget["total_model_calls"],
        )
        if child_role == "qa":
            root = self._root_snapshot()
            budget_data = root.get("budget", {}) or asdict(self._profile_budget())
            reserve = root.get("qa_reserve", {})
            consumed = root.get("consumed", {})
            reservations = root.get("reservations", {})
            reserved_tokens = sum(int((item or {}).get("total_tokens", 0)) for item in reservations.values())
            reserved_calls = sum(int((item or {}).get("model_calls", 0)) for item in reservations.values())
            total_tokens = int(budget_data.get("total_tokens", self._profile_budget().total_tokens))
            total_calls = int(budget_data.get("total_model_calls", self._profile_budget().total_model_calls))
            non_qa_token_cap = max(0, total_tokens - int(reserve.get("total_tokens", 0)))
            non_qa_call_cap = max(0, total_calls - int(reserve.get("model_calls", 0)))
            current_tokens = int(consumed.get("total_tokens", 0)) + reserved_tokens
            current_calls = int(consumed.get("model_calls", 0)) + reserved_calls
            allowed = current_tokens + child_budget["total_tokens"] <= total_tokens and current_calls + child_budget["total_model_calls"] <= total_calls
            if current_tokens < non_qa_token_cap and current_calls < non_qa_call_cap:
                allowed = allowed
        if allowed:
            self._consume(delegation_count=1)
        payload = {
            "child_task_id": child_task_id,
            "engagement_id": self.engagement_id,
            "request_class": self.request_class,
            "profile_key": self.profile_key,
            "selected_agents": selected_agents or [],
            "governance_role": child_role,
            "governance_role_trusted": True,
            "requested_profile": requested_profile or self.profile_key,
            "allocated_budget": child_budget,
            "budget_override": dict(child_budget),
            "allowed": allowed,
        }
        self.log_event("child_budget", payload)
        return payload

    def persist_checkpoint(self, label: str, payload: dict[str, Any]) -> str:
        filename = f"checkpoints/{int(time.time())}-{re.sub(r'[^a-zA-Z0-9_-]+', '-', label)[:40]}.json"
        path = self._persist_json(filename, {
            "timestamp": _utc_now(),
            "label": label,
            "engagement_id": self.engagement_id,
            "task_id": self.task_id,
            "preserved_brief": self._brief_preservation_snapshot(),
            **project_artifact_value(payload),
        })
        if self.state is not None:
            self.state.checkpoints.append(path)
        self.log_event("checkpoint", {"label": label, "path": path})
        return path

    def record_compaction_event(self, *, reason: str, pre_messages: list[dict], post_messages: list[dict], system_message: str = "") -> str:
        self.record_context_manifest(post_messages, system_message=system_message)
        checkpoint = self.persist_checkpoint("context-compaction", {
            "reason": reason,
            "pre_message_count": len(pre_messages),
            "post_message_count": len(post_messages),
            "pre_context_tokens": estimate_messages_tokens(pre_messages),
            "post_context_tokens": estimate_messages_tokens(post_messages),
            "post_context_manifest": self._context_manifest,
        })
        self.log_event("context_compaction", {
            "reason": reason,
            "checkpoint": checkpoint,
            "post_context_manifest": self._context_manifest,
        })
        return checkpoint

    def persist_partial_handoff(
        self,
        *,
        task: str,
        status: str,
        summary: str | None = None,
        errors: Optional[list[Any]] = None,
        limitations: Optional[list[str]] = None,
        artifact_references: Optional[list[str]] = None,
        usage: Optional[dict[str, Any]] = None,
        open_questions: Optional[list[str]] = None,
        recommended_next_step: str = "",
    ) -> dict[str, Any]:
        resume_identifier = f"{self.engagement_id}:{self.task_id}:{uuid.uuid4().hex[:8]}"
        payload = {
            "status": status,
            "task": task,
            "work_completed": [summary] if summary else [],
            "evidence_collected": self._brief_preservation_snapshot().get("evidence_ids", []),
            "claims_evaluated": self._brief_preservation_snapshot().get("claim_ids", []),
            "open_questions": open_questions or self._brief_preservation_snapshot().get("open_questions", []),
            "errors": errors or [],
            "limitations": limitations or [],
            "recommended_next_step": recommended_next_step,
            "artifact_references": artifact_references or [],
            "usage": usage or (asdict(self.state.usage) if self.state else {}),
            "termination_reason": status,
            "resume_token": resume_identifier,
            "resume_identifier": resume_identifier,
            "checkpoints": list(self.state.checkpoints) if self.state else [],
            "context_manifest": list(self._context_manifest),
            "qa_obligations": self._brief_preservation_snapshot().get("qa_obligations", []),
        }
        path = self._persist_json(f"partial_handoffs/{self.task_id}-{int(time.time())}.json", project_artifact_value(payload))
        payload["artifact_references"].append(path)
        self.log_event("partial_handoff", {"status": status, "path": path, "resume_identifier": resume_identifier})
        return payload

    def load_partial_handoff(self, resume_identifier: str) -> dict[str, Any]:
        handoff_dir = self._engagement_dir() / "partial_handoffs"
        for candidate in sorted(handoff_dir.glob("*.json"), reverse=True):
            payload = self._read_json(candidate, default={})
            if payload.get("resume_identifier") == resume_identifier or payload.get("resume_token") == resume_identifier:
                payload["preserved_brief"] = self._brief_preservation_snapshot()
                return payload
        raise FileNotFoundError(f"Unknown resume identifier: {resume_identifier}")

    def close_turn(self, *, status: str, final_response: str = "", termination_reason: str = "completed") -> dict[str, Any]:
        if not self.is_active():
            return {
                "status": status,
                "mode": self.mode,
                "governance_status_path": self._disabled_status_path,
            }
        self.record_tool_schema_metrics(
            getattr(self.agent, "tools", []) or [],
            self._tools_used,
            total_available_count=self._tool_schema_metrics.get("total_available"),
        )
        usage = asdict(self.state.usage) if self.state else {}
        summary = {
            "timestamp": _utc_now(),
            "engagement_id": self.engagement_id,
            "task_id": self.task_id,
            "agent_role": self.role,
            "status": status,
            "mode": self.mode,
            "termination_reason": termination_reason,
            "usage": usage,
            "tool_overhead": {
                "total_available": self._tool_schema_metrics.get("total_available", len(getattr(self.agent, "tools", []) or [])),
                "definitions_sent": self._tool_schema_metrics.get("definitions_sent", len(getattr(self.agent, "tools", []) or [])),
                "serialized_schema_size": self._tool_schema_metrics.get("serialized_schema_size", 0),
                "tools_included": self._tool_schema_metrics.get("tools_included", []),
                "tools_used": sorted(self._tools_used),
                "unused_tools": max(0, self._tool_schema_metrics.get("unused_tools_included", 0)),
            },
            "final_response_chars": len(final_response or ""),
        }
        self._persist_json("summary.json", summary)
        self.log_event("turn_close", summary)
        if self.state is not None:
            self.state.status = status
            self.state.updated_at = _utc_now()
        return summary


def build_governance_controller(agent: Any, config_mapping: Optional[dict]) -> GovernanceController:
    return GovernanceController(agent, GovernanceConfig.from_mapping(config_mapping))
