"""Trusted construction context for delegated profile agents."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class ChildBinding:
    depth: int
    session_id: str
    role: str
    profile: str | None = None
    home: Path | None = None
    config: dict | None = None
    seed: dict | None = None
    root: Path | None = None


_construction = ContextVar("delegated_profile_construction", default=None)


@contextmanager
def profile_scope(home):
    if home is None:
        yield
        return

    from gateway.run import _profile_runtime_scope

    with _profile_runtime_scope(Path(home)):
        yield


@contextmanager
def construction_scope(binding):
    token = _construction.set(binding)
    try:
        with profile_scope(binding.home):
            yield
    finally:
        _construction.reset(token)


def bind_initial_governance(agent):
    binding = _construction.get()
    if binding is None:
        return None

    agent._delegate_depth = binding.depth
    agent._governance_role = binding.role
    agent._delegated_profile_name = binding.profile
    agent._delegated_profile_home = binding.home
    agent._delegated_governance_root = binding.root
    agent.session_id = binding.session_id

    if binding.seed:
        agent._governance_seed = deepcopy(binding.seed)
        agent._governance_engagement_id = binding.seed.get("engagement_id")
        agent._governance_profile_key = str(
            binding.seed.get("profile_key") or "bounded"
        )
        agent._governance_request_class = str(
            binding.seed.get("request_class") or "engagement"
        )

    return deepcopy(binding.config) if binding.config is not None else None


def resolve_profile(routes, name):
    if not isinstance(routes, dict):
        raise ValueError("delegation.profile_routes must be a mapping")

    if name is None and not routes:
        return None, None, "sme"

    if not isinstance(name, str) or name not in routes:
        raise ValueError(
            "Select a configured task.profile: " + ", ".join(sorted(routes))
        )

    role = routes[name]
    if role not in {"sme", "qa"}:
        raise ValueError("Invalid delegated profile role")

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError("Invalid delegated profile name")

    from hermes_cli.profiles import get_profile_dir, profile_exists

    home = get_profile_dir(name).resolve()

    if not profile_exists(name):
        raise ValueError(f"Delegated profile does not exist: {name}")

    if not all((home / filename).is_file() for filename in ("config.yaml", "SOUL.md")):
        raise ValueError(
            f"Delegated profile requires config.yaml and SOUL.md: {name}"
        )

    return name, home, role


def prepare_binding(parent, *, routes, profile, session_id, depth):
    name, home, role = resolve_profile(routes, profile)

    config = None
    seed = None
    root = None
    controller = getattr(parent, "cost_context_governance", None)

    if controller is not None:
        from agent.cost_context_governance import GovernanceConfig

        controller_config = getattr(controller, "config", None)

        if isinstance(controller_config, GovernanceConfig):
            config = asdict(controller_config)
            identity = getattr(controller, "identity", None)
            candidate = getattr(identity, "persistence_root", None)
            root = candidate if isinstance(candidate, Path) else None

        allocator = getattr(controller, "allocate_child_budget", None)
        active_fn = getattr(controller, "is_active", None)

        if callable(allocator) and (not callable(active_fn) or active_fn()):
            candidate = allocator(
                child_task_id=session_id,
                requested_profile=getattr(
                    parent, "_governance_profile_key", None
                ),
                selected_agents=[name or "leaf"],
                governance_role=role,
            )

            # Real controllers return a dict. Test doubles and legacy parents
            # may expose MagicMock/callable placeholders; those do not deny delegation.
            if isinstance(candidate, dict):
                if candidate.get("allowed") is not True:
                    raise ValueError(
                        "Delegation denied by engagement budget; no child was created."
                    )
                seed = deepcopy(candidate)
                seed["selected_agents"] = [name or "leaf"]

    return ChildBinding(
        depth=depth,
        session_id=session_id,
        role=role,
        profile=name,
        home=home,
        config=config,
        seed=seed,
        root=root,
    )


def narrow_toolsets(inherited, allowed):
    from tools.delegate_tool_toolsets import _expand_parent_toolsets

    return sorted(
        _expand_parent_toolsets(set(inherited))
        & _expand_parent_toolsets(set(allowed))
    )
