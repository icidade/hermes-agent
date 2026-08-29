from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
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
    profiles: dict[str, BudgetProfile] = field(default_factory=lambda: {
        "conversational": BudgetProfile(
            total_model_calls=3,
            calls_per_agent=3,
            calls_per_delegated_task=0,
            input_tokens=6000,
            output_tokens=3000,
            total_tokens=9000,
            context_tokens_per_request=5000,
            wall_clock_seconds=90,
            tool_invocations=6,
            retries=1,
            delegation_count=0,
            iterations_without_progress=2,
            qa_reserve_ratio=0.0,
        ),
        "bounded": BudgetProfile(
            total_model_calls=6,
            calls_per_agent=5,
            calls_per_delegated_task=3,
            input_tokens=18000,
            output_tokens=8000,
            total_tokens=26000,
            context_tokens_per_request=9000,
            wall_clock_seconds=180,
            tool_invocations=15,
            retries=3,
            delegation_count=1,
            iterations_without_progress=3,
            qa_reserve_ratio=0.10,
        ),
        "standard_engagement": BudgetProfile(),
        "deep_engagement": BudgetProfile(
            total_model_calls=16,
            calls_per_agent=10,
            calls_per_delegated_task=8,
            input_tokens=60000,
            output_tokens=24000,
            total_tokens=84000,
            context_tokens_per_request=18000,
            wall_clock_seconds=600,
            tool_invocations=50,
            retries=6,
            delegation_count=4,
            iterations_without_progress=5,
            qa_reserve_ratio=0.25,
        ),
    })
    role_toolsets: dict[str, list[str]] = field(default_factory=lambda: {
        "chief": ["delegation", "todo", "session_search", "file", "search", "skills", "browser", "web"],
        "sme": ["file", "search", "terminal", "web", "browser"],
        "qa": ["file", "search", "terminal", "web", "browser", "session_search"],
    })

    @classmethod
    def from_mapping(cls, data: Optional[dict]) -> "GovernanceConfig":
        raw = data or {}
        cfg = cls()
        cfg.mode = str(raw.get("mode", cfg.mode))
        cfg.workspace_dir = str(raw.get("workspace_dir", cfg.workspace_dir))
        cfg.default_profile = str(raw.get("default_profile", cfg.default_profile))
        cfg.force_for_profiles = list(raw.get("force_for_profiles", cfg.force_for_profiles) or [])
        cfg.engagement_keywords = list(raw.get("engagement_keywords", cfg.engagement_keywords) or [])
        cfg.role_toolsets = dict(raw.get("role_toolsets", cfg.role_toolsets) or {})
        threshold = raw.get("threshold") or {}
        cfg.threshold = ThresholdConfig(
            informational_ratio=_safe_float(threshold.get("informational_ratio"), cfg.threshold.informational_ratio),
            warning_ratio=_safe_float(threshold.get("warning_ratio"), cfg.threshold.warning_ratio),
            approval_ratio=_safe_float(threshold.get("approval_ratio"), cfg.threshold.approval_ratio),
            hard_stop_ratio=_safe_float(threshold.get("hard_stop_ratio"), cfg.threshold.hard_stop_ratio),
        )
        progress = raw.get("progress") or {}
        cfg.progress = ProgressConfig(
            equivalent_tool_calls=_safe_int(progress.get("equivalent_tool_calls"), cfg.progress.equivalent_tool_calls),
            same_error_repeats=_safe_int(progress.get("same_error_repeats"), cfg.progress.same_error_repeats),
            no_progress_iterations=_safe_int(progress.get("no_progress_iterations"), cfg.progress.no_progress_iterations),
            timeout_checkpoint_seconds=_safe_int(progress.get("timeout_checkpoint_seconds"), cfg.progress.timeout_checkpoint_seconds),
        )
        profiles = dict(cfg.profiles)
        for name, profile_data in (raw.get("profiles") or {}).items():
            base = profiles.get(name, BudgetProfile())
            profiles[name] = BudgetProfile.from_mapping(profile_data, base)
        cfg.profiles = profiles
        return cfg


@dataclass
class EnvelopeUsage:
    model_calls: int = 0
    tool_invocations: int = 0
    delegation_count: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    context_tokens_max: int = 0
    no_progress_iterations: int = 0
    progress_events: int = 0


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


class FileBackedBudget:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def update(self, mutator):
        with self.path.open("r+", encoding="utf-8") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            raw = fh.read().strip() or "{}"
            data = json.loads(raw)
            new_data = mutator(data) or data
            fh.seek(0)
            fh.truncate(0)
            fh.write(_json_dumps(new_data))
            fh.flush()
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return new_data


class GovernanceController:
    def __init__(self, agent: Any, config: GovernanceConfig):
        self.agent = agent
        self.config = config
        self.mode = config.mode
        self.profile_name = self._discover_profile_name()
        self.role = self._discover_role()
        self.root_dir = self._resolve_workspace_dir()
        self.engagement_id = getattr(agent, "_governance_engagement_id", None)
        self.task_id = getattr(agent, "session_id", None) or uuid.uuid4().hex[:12]
        self.request_class = getattr(agent, "_governance_request_class", "conversational")
        self.profile_key = getattr(agent, "_governance_profile_key", config.default_profile)
        self.state: EnvelopeState | None = None
        self._budget_file: FileBackedBudget | None = None
        self._last_tool_signature: str | None = None
        self._last_error_signature: str | None = None
        self._tools_used: set[str] = set()
        self._context_manifest: list[str] = []
        self._turn_started_at = time.time()
        self._initialized = False

    def _discover_profile_name(self) -> str:
        env_profile = os.environ.get("HERMES_PROFILE") or os.environ.get("HERMES_ACTIVE_PROFILE")
        if env_profile:
            return env_profile
        config_path = os.environ.get("HERMES_CONFIG_PATH", "")
        m = re.search(r"/profiles/([^/]+)/config\.yaml$", config_path)
        if m:
            return m.group(1)
        return "default"

    def _discover_role(self) -> str:
        role = getattr(self.agent, "_governance_role", None)
        if role:
            return role
        if getattr(self.agent, "_delegate_depth", 0) == 0:
            return "chief"
        return "sme"

    def _resolve_workspace_dir(self) -> Path:
        cfg_path = os.environ.get("HERMES_CONFIG_PATH")
        if cfg_path and "/profiles/" in cfg_path:
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

    def is_active(self) -> bool:
        return self.mode in {"observe", "enforce"}

    def should_enforce(self) -> bool:
        return self.mode == "enforce"

    def _profile_budget(self) -> BudgetProfile:
        return self.config.profiles.get(self.profile_key, self.config.profiles[self.config.default_profile])

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
        if self.mode == "disabled":
            return
        seed = getattr(self.agent, "_governance_seed", None)
        self.request_class, self.profile_key = self.classify(user_message)
        if getattr(self.agent, "_governance_profile_key", None):
            self.profile_key = self.agent._governance_profile_key
        if getattr(self.agent, "_governance_request_class", None):
            self.request_class = self.agent._governance_request_class
        if isinstance(seed, dict) and seed.get("engagement_id"):
            self.engagement_id = str(seed.get("engagement_id"))
            self.profile_key = str(seed.get("profile_key") or self.profile_key)
            self.request_class = str(seed.get("request_class") or self.request_class)
        self.engagement_id = self.engagement_id or f"eng-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
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
            objective=(user_message or "")[:500],
            selected_agents=[],
            qa_reserve={"total_tokens": qa_tokens, "model_calls": qa_calls},
            budget=asdict(budget),
        )
        self._budget_file = FileBackedBudget(self.root_dir / self.engagement_id / "budget.json")
        if not self._initialized:
            self._budget_file.update(lambda data: self._bootstrap_budget(data, budget))
            self._initialized = True
        self._persist_json("brief.json", {
            "engagement_id": self.engagement_id,
            "request_class": self.request_class,
            "objective": user_message,
            "scope": [],
            "constraints": [],
            "selected_agents": [],
            "resource_envelope": asdict(budget),
            "qa_reserve": self.state.qa_reserve,
            "status": self.state.status,
            "artifact_references": [],
            "evidence_ledger": "evidence_ledger.jsonl",
            "claim_ledger": "claim_ledger.jsonl",
            "open_questions": [],
            "checkpoints": [],
            "usage_summary": asdict(self.state.usage),
        })
        self.record_context_manifest(messages, system_message=system_message)
        self.log_event("turn_begin", {
            "request_class": self.request_class,
            "profile": self.profile_key,
            "session_id": getattr(self.agent, "session_id", None),
            "agent_role": self.role,
            "selected_agents": list(seed.get("selected_agents") or []) if isinstance(seed, dict) else [],
        })
        self.agent._governance_engagement_id = self.engagement_id
        self.agent._governance_request_class = self.request_class
        self.agent._governance_profile_key = self.profile_key
        self.agent._governance_role = self.role

    def _bootstrap_budget(self, data: dict, budget: BudgetProfile) -> dict:
        if data.get("engagement_id"):
            return data
        return {
            "engagement_id": self.engagement_id,
            "budget": asdict(budget),
            "qa_reserve": {
                "total_tokens": int(budget.total_tokens * float(budget.qa_reserve_ratio or 0.0)),
                "model_calls": int(budget.total_model_calls * float(budget.qa_reserve_ratio or 0.0)),
            },
            "consumed": asdict(EnvelopeUsage()),
            "children": {},
        }

    def _engagement_dir(self) -> Path:
        return self.root_dir / self.engagement_id

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
        self._append_jsonl("telemetry.jsonl", record)

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

    def record_tool_schema_metrics(self, tool_defs: list[dict], tools_used: Optional[Iterable[str]] = None) -> None:
        serialized = _json_dumps(tool_defs or [])
        used = set(tools_used or [])
        count = len(tool_defs or [])
        unused = max(0, count - len(used & {t.get("function", {}).get("name") for t in tool_defs or []}))
        self.log_event("tool_schema", {
            "tool_definition_count": count,
            "tool_schema_serialized_size": len(serialized.encode("utf-8")),
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
            return data

        updated = budget_file.update(mutate)
        if self.state is not None:
            for key, value in delta.items():
                if hasattr(self.state.usage, key):
                    setattr(self.state.usage, key, int(getattr(self.state.usage, key)) + int(value))
            self.state.updated_at = _utc_now()
        return updated

    def _root_consumed(self) -> dict[str, int]:
        if self._budget_file is None:
            return {}
        data = self._budget_file.update(lambda current: current)
        return data.get("consumed", {})

    def _qa_headroom_violation(self, projected_total_tokens: int = 0, projected_model_calls: int = 0) -> bool:
        if self.role == "qa":
            return False
        root = self._root_consumed()
        budget = self._profile_budget()
        reserve_tokens = int(budget.total_tokens * float(budget.qa_reserve_ratio or 0.0))
        reserve_calls = int(budget.total_model_calls * float(budget.qa_reserve_ratio or 0.0))
        remaining_tokens = int(budget.total_tokens) - int(root.get("total_tokens", 0)) - int(projected_total_tokens)
        remaining_calls = int(budget.total_model_calls) - int(root.get("model_calls", 0)) - int(projected_model_calls)
        return remaining_tokens < reserve_tokens or remaining_calls < reserve_calls

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

    def before_model_call(self, *, messages: list[dict], approx_request_tokens: int, api_call_count: int) -> dict[str, Any]:
        if not self.is_active():
            return {"action": "allow"}
        budget = self._profile_budget()
        context_tokens = estimate_messages_tokens(messages)
        self.record_context_manifest(messages)
        self.state.usage.context_tokens_max = max(self.state.usage.context_tokens_max, context_tokens) if self.state else context_tokens
        ratios = [
            approx_request_tokens / max(1, int(budget.context_tokens_per_request)),
            (api_call_count + 1) / max(1, int(budget.calls_per_agent)),
            (time.time() - self._turn_started_at) / max(1, int(budget.wall_clock_seconds)),
        ]
        action = self._threshold_action(max(ratios))
        if self._qa_headroom_violation(projected_total_tokens=approx_request_tokens, projected_model_calls=1):
            action = "approval" if action != "hard_stop" else action
        payload = {
            "action": action,
            "context_tokens": context_tokens,
            "approx_request_tokens": approx_request_tokens,
            "api_call_count": api_call_count,
        }
        if action != "allow":
            self.log_event("threshold_event", payload)
        if action == "warning":
            return {**payload, "compact_context": True}
        if action == "approval":
            return {**payload, "pause": True, "message": "Governança exigiu checkpoint/aprovação antes de continuar."}
        if action == "hard_stop" and self.should_enforce():
            self.persist_checkpoint("hard-stop-before-model", {
                "messages_tail": messages[-6:],
                "approx_request_tokens": approx_request_tokens,
            })
            return {**payload, "stop": True, "message": "Hard stop de custo/contexto atingido antes da próxima chamada ao modelo."}
        return payload

    def record_model_usage(self, usage: Any | None, *, duration_seconds: float, retry_count: int = 0, approx_request_tokens: int = 0, response_text: str = "") -> dict[str, Any]:
        if not self.is_active():
            return {}
        canonical = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "cache_creation_tokens": 0,
            "reasoning_tokens": 0,
            "estimated": False,
        }
        if usage is not None:
            for key in list(canonical.keys()):
                if key == "estimated":
                    continue
                canonical[key] = _safe_int(getattr(usage, key, None), canonical[key])
            if not canonical["input_tokens"] and hasattr(usage, "prompt_tokens"):
                canonical["input_tokens"] = _safe_int(getattr(usage, "prompt_tokens", 0), 0)
            if not canonical["output_tokens"] and hasattr(usage, "completion_tokens"):
                canonical["output_tokens"] = _safe_int(getattr(usage, "completion_tokens", 0), 0)
            if not canonical["total_tokens"]:
                canonical["total_tokens"] = canonical["input_tokens"] + canonical["output_tokens"]
        if canonical["total_tokens"] <= 0:
            canonical["estimated"] = True
            canonical["input_tokens"] = int(approx_request_tokens * 1.15)
            canonical["output_tokens"] = max(32, int(estimate_text_tokens(response_text) * 1.20))
            canonical["total_tokens"] = canonical["input_tokens"] + canonical["output_tokens"]
        self._consume(
            model_calls=1,
            retries=retry_count,
            input_tokens=canonical["input_tokens"],
            output_tokens=canonical["output_tokens"],
            total_tokens=canonical["total_tokens"],
        )
        self.log_event("model_call", {
            "provider": getattr(self.agent, "provider", None),
            "model": getattr(self.agent, "model", None),
            "request_class": self.request_class,
            "duration_seconds": duration_seconds,
            "input_tokens": canonical["input_tokens"],
            "output_tokens": canonical["output_tokens"],
            "total_tokens": canonical["total_tokens"],
            "cached_input_tokens": canonical["cached_input_tokens"],
            "cache_creation_tokens": canonical["cache_creation_tokens"],
            "reasoning_tokens": canonical["reasoning_tokens"],
            "estimated_tokens": canonical["estimated"],
            "context_manifest": self._context_manifest,
            "retry_count": retry_count,
        })
        return canonical

    def before_tool_call(self, tool_name: str, tool_args: dict) -> dict[str, Any]:
        if not self.is_active():
            return {"action": "allow"}
        budget = self._profile_budget()
        projected = (self.state.usage.tool_invocations + 1) if (self.state is not None) else 1
        action = self._threshold_action(projected / max(1, int(budget.tool_invocations)))
        signature = hash_context_block({"tool": tool_name, "args": tool_args})
        if signature == self._last_tool_signature:
            next_no_progress = (self.state.usage.no_progress_iterations + 1) if self.state else 1
            if next_no_progress >= int(budget.iterations_without_progress):
                action = "hard_stop" if self.should_enforce() else "warning"
        self._last_tool_signature = signature
        if action in {"warning", "approval", "hard_stop"}:
            self.log_event("tool_threshold", {
                "tool_name": tool_name,
                "action": action,
                "tool_invocations_projected": projected,
            })
        if action == "hard_stop" and self.should_enforce():
            self.persist_checkpoint("hard-stop-before-tool", {
                "tool_name": tool_name,
                "tool_args": tool_args,
            })
            return {"action": "hard_stop", "message": f"Governança bloqueou nova chamada para {tool_name}."}
        return {"action": action}

    def record_tool_result(self, tool_name: str, tool_args: dict, result: str, *, is_error: bool, duration_seconds: float) -> None:
        if not self.is_active():
            return
        self._tools_used.add(tool_name)
        no_progress_delta = 0
        progress_delta = 0
        error_signature = hash_context_block({"tool": tool_name, "result": result[:1000] if result else "", "error": is_error})
        if is_error:
            if error_signature == self._last_error_signature:
                no_progress_delta = 1
            self._last_error_signature = error_signature
        else:
            progress_delta = 1 if (result and str(result).strip()) else 0
            no_progress_delta = 0 if progress_delta else 1
            if progress_delta:
                self._last_error_signature = None
        self._consume(tool_invocations=1, no_progress_iterations=no_progress_delta, progress_events=progress_delta)
        self.log_event("tool_call", {
            "tool_name": tool_name,
            "duration_seconds": duration_seconds,
            "error": bool(is_error),
            "progress_delta": progress_delta,
            "no_progress_delta": no_progress_delta,
            "args_hash": hash_context_block(tool_args),
        })

    def allocate_child_budget(self, *, child_task_id: str, requested_profile: str | None = None, selected_agents: Optional[list[str]] = None) -> dict[str, Any]:
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
        if self._qa_headroom_violation(projected_total_tokens=child_budget["total_tokens"], projected_model_calls=child_budget["total_model_calls"]):
            allowed = False
        else:
            allowed = True
            self._consume(delegation_count=1)
        payload = {
            "child_task_id": child_task_id,
            "engagement_id": self.engagement_id,
            "request_class": self.request_class,
            "profile_key": self.profile_key,
            "selected_agents": selected_agents or [],
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
            **payload,
        })
        if self.state is not None:
            self.state.checkpoints.append(path)
        self.log_event("checkpoint", {"label": label, "path": path})
        return path

    def persist_partial_handoff(self, *, task: str, status: str, summary: str | None = None, errors: Optional[list[str]] = None, limitations: Optional[list[str]] = None, artifact_references: Optional[list[str]] = None, usage: Optional[dict[str, Any]] = None, open_questions: Optional[list[str]] = None, recommended_next_step: str = "") -> dict[str, Any]:
        payload = {
            "status": status,
            "task": task,
            "work_completed": [summary] if summary else [],
            "evidence_collected": [],
            "claims_evaluated": [],
            "open_questions": open_questions or [],
            "errors": errors or [],
            "limitations": limitations or [],
            "recommended_next_step": recommended_next_step,
            "artifact_references": artifact_references or [],
            "usage": usage or (asdict(self.state.usage) if self.state else {}),
            "resume_token": f"{self.engagement_id}:{self.task_id}:{uuid.uuid4().hex[:8]}",
        }
        path = self._persist_json(f"partial_handoffs/{self.task_id}-{int(time.time())}.json", payload)
        payload["artifact_references"].append(path)
        self.log_event("partial_handoff", {"status": status, "path": path})
        return payload

    def close_turn(self, *, status: str, final_response: str = "", termination_reason: str = "completed") -> dict[str, Any]:
        if not self.is_active():
            return {}
        self.record_tool_schema_metrics(getattr(self.agent, "tools", []) or [], self._tools_used)
        usage = asdict(self.state.usage) if self.state else {}
        summary = {
            "timestamp": _utc_now(),
            "engagement_id": self.engagement_id,
            "task_id": self.task_id,
            "agent_role": self.role,
            "status": status,
            "termination_reason": termination_reason,
            "usage": usage,
            "tool_overhead": {
                "definitions_sent": len(getattr(self.agent, "tools", []) or []),
                "tools_used": sorted(self._tools_used),
                "unused_tools": max(0, len(getattr(self.agent, "tools", []) or []) - len(self._tools_used)),
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
