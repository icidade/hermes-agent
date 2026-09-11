import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.cost_context_governance import (
    BudgetProfile,
    FileBackedBudget,
    GovernanceController,
    GovernanceConfig,
)


def _make_agent(depth=0, *, role=None, session_id="sess-123"):
    agent = MagicMock()
    agent._delegate_depth = depth
    agent._governance_engagement_id = None
    agent._governance_request_class = None
    agent._governance_profile_key = None
    agent._governance_seed = None
    agent.provider = "openai-codex"
    agent.model = "gpt-5.4"
    agent.session_id = session_id
    agent.tools = []
    agent.profile = "chief-of-staff"
    agent.platform = "telegram"
    agent._user_id = "owner-123"
    agent._chat_id = "chat-123"
    agent._governance_owner_identity_trusted = True
    agent._governance_callback_capability = object()
    agent._governance_role = role or ("chief" if depth == 0 else "sme")
    return agent


def _owner_principal(*, profile="chief-of-staff", engagement_id=None, subject="owner-123", chat_id="chat-123", task_id="auth-valid"):
    return {
        "authenticated": True,
        "authentication_source": "gateway_adapter",
        "subject": subject,
        "platform": "telegram",
        "profile": profile,
        "engagement_id": engagement_id,
        "chat_id": chat_id,
        "task_id": task_id,
    }


def _explicit_increment(tokens=1000, calls=1, per_request=800):
    return {
        "operation": "increment",
        "total_tokens": tokens,
        "total_model_calls": calls,
        "context_tokens_per_request": per_request,
    }


def _make_controller(tmp_path, monkeypatch, *, mode="observe", profile_name="chief-of-staff", depth=0, role=None, session_id="sess-123"):
    cfg_dir = tmp_path / "profiles" / profile_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_CONFIG_PATH", str(cfg_dir / "config.yaml"))
    monkeypatch.setenv("HERMES_PROFILE", profile_name)
    config = GovernanceConfig.from_mapping(
        {
            "mode": mode,
            "workspace_dir": str(tmp_path / "engagements"),
            "force_for_profiles": [profile_name],
        }
    )
    agent = _make_agent(depth=depth, role=role, session_id=session_id)
    controller = GovernanceController(agent, config)
    return controller, agent


def _prepare_dispatched_request(controller: GovernanceController, request_id: str, estimate: int = 80) -> str:
    controller.reserve_request(request_id, estimate)
    controller.record_dispatch(request_id)
    return request_id


def test_file_backed_budget_distinguishes_missing_from_invalid_json(tmp_path):
    path = tmp_path / "budget.json"
    budget = FileBackedBudget(path)

    assert not path.exists()
    assert budget.initialize({"initialized": True})["initialized"] is True

    path.write_text("{\"initialized\":", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid governance artifact"):
        budget.update(lambda data: data)


@pytest.mark.parametrize("contents", ["", "{}", "[]", "null"])
def test_existing_budget_artifact_is_not_treated_as_missing(tmp_path, contents):
    path = tmp_path / "budget.json"
    path.write_text(contents, encoding="utf-8")
    budget = FileBackedBudget(path)

    with pytest.raises(RuntimeError, match="invalid governance artifact"):
        budget.update(lambda data: data)


def test_budget_missing_field_and_invalid_decisional_type_fail_closed(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="MiniCISO security review", system_message="system", messages=[], task_id="shape")
    assert controller._budget_file is not None
    path = controller._budget_file.path
    baseline = json.loads(path.read_text(encoding="utf-8"))
    for mutation in (
        lambda data: data.pop("budget_authorizations"),
        lambda data: data.__setitem__("consumed", []),
    ):
        data = json.loads(json.dumps(baseline))
        mutation(data)
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(RuntimeError, match="invalid governance artifact"):
            controller._budget_file.update(lambda current: current)


def test_budget_read_and_persist_failures_hide_exception_content(tmp_path, monkeypatch):
    path = tmp_path / "budget.json"
    budget = FileBackedBudget(path)
    budget.initialize({
        "engagement_id": "eng-test", "budget": {}, "qa_reserve": {}, "consumed": {},
        "children": {}, "reservations": {}, "request_state": {}, "logical_calls": {},
        "budget_authorizations": {}, "accounting_latch": {},
    })
    canary = "SECRET_CANARY /home/private/governance.json"
    with patch.object(Path, "read_text", side_effect=OSError(canary)):
        with pytest.raises(RuntimeError, match="invalid governance artifact") as read_error:
            budget.update(lambda data: data)
    assert read_error.value.args == ("invalid governance artifact",)

    with patch.object(Path, "replace", side_effect=OSError(canary)):
        with pytest.raises(RuntimeError, match="invalid governance artifact") as persist_error:
            budget.update(lambda data: data)
    assert persist_error.value.args == ("invalid governance artifact",)


def test_enforce_invalid_budget_blocks_before_provider_and_tools(tmp_path, monkeypatch):
    controller, agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    agent.client = MagicMock()
    controller.begin_turn(
        user_message="Preciso de uma análise MiniCISO de segurança",
        system_message="system",
        messages=[],
        task_id="invalid-budget",
    )
    assert controller._budget_file is not None
    budget_path = controller._budget_file.path
    budget_path.write_text("{\"consumed\":", encoding="utf-8")

    decision = controller.before_model_call(
        request_id="invalid-budget:req:1",
        messages=[{"role": "user", "content": "canary"}],
        approx_request_tokens=10,
        api_call_count=0,
    )
    assert decision["allowed"] is False
    assert decision["reason"] == "invalid_governance_artifact"
    assert decision["message"] == "Governança bloqueou a execução porque um artifact decisório está inválido."
    errors_path = budget_path.parent / "governance_artifact_errors.jsonl"
    assert errors_path.exists()
    assert "canary" not in errors_path.read_text(encoding="utf-8").lower()
    assert agent.client.chat.completions.create.call_count == 0


def test_observe_invalid_budget_remains_observable_without_enforcing(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="observe")
    controller.begin_turn(user_message="review", system_message="system", messages=[], task_id="observe-invalid")
    assert controller._budget_file is not None
    controller._budget_file.path.write_text("{\"consumed\":", encoding="utf-8")

    decision = controller.before_model_call(
        request_id="observe-invalid:req:1",
        messages=[{"role": "user", "content": "continue"}],
        approx_request_tokens=10,
        api_call_count=0,
    )

    assert decision["allowed"] is True
    assert decision["action"] == "informational"
    assert decision["reason"] == "invalid_governance_artifact_observed"
    assert (controller._budget_file.path.parent / "governance_artifact_errors.jsonl").exists()


def test_malformed_authorization_record_cannot_be_consumed(tmp_path, monkeypatch):
    controller, agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(
        user_message="Preciso de uma análise MiniCISO de segurança",
        system_message="system",
        messages=[],
        task_id="invalid-auth",
    )
    assert controller._budget_file is not None
    data = json.loads(controller._budget_file.path.read_text(encoding="utf-8"))
    data["budget_authorizations"] = {
        "auth-invalid": {
            "authorization_id": "auth-invalid",
            "status": "authorized",
            "engagement_id": controller.engagement_id,
            "profile": controller.profile_name,
            "task_id": controller.task_id,
            "expires_at": "2099-01-01T00:00:00+00:00",
            "approved_envelope": "not-an-object",
        }
    }
    controller._budget_file.path.write_text(json.dumps(data), encoding="utf-8")
    agent._governance_resume_authorization_id = "auth-invalid"
    agent._governance_resume_logical_call_id = "logical-1"

    decision = controller.before_model_call(
        request_id="invalid-auth:req:1",
        messages=[{"role": "user", "content": "canary"}],
        approx_request_tokens=10,
        api_call_count=0,
        logical_call_id="logical-1",
    )
    assert decision["allowed"] is False
    assert decision["reason"] == "invalid_governance_artifact"
    assert agent.client.chat.completions.create.call_count == 0


def _process_increment_budget(path_str: str, increments: int, queue: multiprocessing.Queue) -> None:
    budget = FileBackedBudget(Path(path_str))
    for _ in range(increments):
        budget.update(
            lambda data: {
                **data,
                "consumed": {
                    **(data.get("consumed") or {}),
                    "model_calls": int((data.get("consumed") or {}).get("model_calls", 0)) + 1,
                },
            }
        )
    queue.put(True)


def _process_child_consume(
    config_mapping: dict,
    profile_name: str,
    workspace_root: str,
    engagement_id: str,
    task_id: str,
    role: str,
    queue: multiprocessing.Queue,
) -> None:
    import os

    os.environ["HERMES_PROFILE"] = profile_name
    os.environ["HERMES_CONFIG_PATH"] = str(Path(workspace_root) / "profiles" / profile_name / "config.yaml")
    agent = SimpleNamespace(
        _delegate_depth=1,
        _governance_engagement_id=engagement_id,
        _governance_request_class="engagement",
        _governance_profile_key="standard_engagement",
        _governance_seed={
            "engagement_id": engagement_id,
            "request_class": "engagement",
            "profile_key": "standard_engagement",
            "budget_override": config_mapping["profiles"]["standard_engagement"],
        },
        provider="openai-codex",
        model="gpt-5.4",
        session_id=task_id,
        tools=[],
        profile=profile_name,
        _governance_role=role,
    )
    controller = GovernanceController(agent, GovernanceConfig.from_mapping(config_mapping))
    controller.begin_turn(user_message="child task", system_message="system", messages=[], task_id=task_id)
    request_id = _prepare_dispatched_request(controller, f"{task_id}:req:1", estimate=500)
    controller.record_model_usage(
        None,
        request_id=request_id,
        duration_seconds=0.1,
        approx_request_tokens=500,
        response_text="evidence collected",
    )
    queue.put(controller._root_consumed())


def test_begin_turn_classifies_miniciso_request_and_writes_brief(tmp_path, monkeypatch):
    controller, agent = _make_controller(tmp_path, monkeypatch)

    controller.begin_turn(
        user_message="Preciso de uma análise MiniCISO de segurança",
        system_message="system",
        messages=[{"role": "user", "content": "oi"}],
        task_id="task-1",
    )

    assert controller.request_class == "engagement"
    assert controller.profile_key == "standard_engagement"
    assert controller.state is not None
    assert controller.state.qa_reserve["total_tokens"] > 0
    assert agent._cost_context_governance_mode == "observe"

    brief = Path(tmp_path / "engagements" / controller.engagement_id / "brief.json")
    assert brief.exists()
    data = json.loads(brief.read_text(encoding="utf-8"))
    assert data["request_class"] == "engagement"
    assert data["resource_envelope"]["total_model_calls"] == BudgetProfile().total_model_calls
    assert data["evidence_ids"] == []
    assert data["claim_ids"] == []


def test_disabled_mode_is_explicit_and_visible(tmp_path, monkeypatch):
    controller, agent = _make_controller(tmp_path, monkeypatch, mode="disabled")
    controller.begin_turn(user_message="oi", system_message="system", messages=[], task_id="task-disabled")
    summary = controller.close_turn(status="completed", final_response="ok")

    status_path = Path(agent._cost_context_governance_status_path)
    assert status_path.exists()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["mode"] == "disabled"
    assert status["active"] is False
    assert summary["mode"] == "disabled"
    decision = controller.before_model_call(
        request_id="disabled:req:1",
        messages=[{"role": "user", "content": "hello"}],
        approx_request_tokens=10,
        api_call_count=0,
    )
    assert decision == {
        "request_id": "disabled:req:1",
        "allowed": True,
        "action": "disabled",
        "reason": "operator_disabled",
        "compact_context": False,
        "termination_reason": None,
    }


def test_before_model_call_requests_pause_when_non_qa_would_enter_qa_reserve(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(
        user_message="MiniCISO security review",
        system_message="system",
        messages=[],
        task_id="task-2",
    )
    assert controller._budget_file is not None
    budget = controller._profile_budget()
    reserve = int(budget.total_tokens * budget.qa_reserve_ratio)

    controller._budget_file.update(
        lambda data: {
            **data,
            "consumed": {
                **data.get("consumed", {}),
                "total_tokens": budget.total_tokens - reserve + 1,
                "model_calls": 0,
            },
        }
    )

    decision = controller.before_model_call(
        request_id="task-2:req:1",
        messages=[{"role": "user", "content": "x" * 200}],
        approx_request_tokens=200,
        api_call_count=0,
    )

    assert decision["action"] == "approval"
    assert decision["pause"] is True


def test_sme_cannot_consume_qa_reserve(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="MiniCISO assessment", system_message="system", messages=[], task_id="root")
    budget = controller._profile_budget()
    reserve = int(budget.total_tokens * budget.qa_reserve_ratio)
    assert controller._budget_file is not None
    controller._budget_file.update(
        lambda data: {
            **data,
            "consumed": {**data.get("consumed", {}), "total_tokens": budget.total_tokens - reserve + 10, "model_calls": 0},
        }
    )

    child_controller, child_agent = _make_controller(tmp_path, monkeypatch, mode="enforce", depth=1, role="sme", session_id="child-sme")
    child_agent._governance_engagement_id = controller.engagement_id
    child_agent._governance_request_class = "engagement"
    child_agent._governance_profile_key = "standard_engagement"
    child_agent._governance_seed = {
        "engagement_id": controller.engagement_id,
        "request_class": "engagement",
        "profile_key": "standard_engagement",
        "budget_override": controller._profile_budget().__dict__.copy(),
        "governance_role": "sme",
        "governance_role_trusted": True,
    }
    child_controller = GovernanceController(child_agent, child_controller.config)
    child_controller.begin_turn(user_message="child", system_message="system", messages=[], task_id="child-sme")
    decision = child_controller.before_model_call(request_id="child-sme:req:1", messages=[{"role": "user", "content": "probe"}], approx_request_tokens=50, api_call_count=0)
    assert decision["action"] in {"approval", "hard_stop"}
    assert decision.get("pause") or decision.get("stop")


def test_qa_can_consume_qa_reserve(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="MiniCISO assessment", system_message="system", messages=[], task_id="root")
    budget = controller._profile_budget()
    reserve = int(budget.total_tokens * budget.qa_reserve_ratio)
    assert controller._budget_file is not None
    controller._budget_file.update(
        lambda data: {
            **data,
            "consumed": {**data.get("consumed", {}), "total_tokens": budget.total_tokens - reserve + 10, "model_calls": 0},
        }
    )

    qa_controller, qa_agent = _make_controller(tmp_path, monkeypatch, mode="enforce", depth=1, role="qa", session_id="child-qa")
    qa_agent._governance_engagement_id = controller.engagement_id
    qa_agent._governance_request_class = "engagement"
    qa_agent._governance_profile_key = "standard_engagement"
    qa_agent._governance_seed = {
        "engagement_id": controller.engagement_id,
        "request_class": "engagement",
        "profile_key": "standard_engagement",
        "budget_override": controller._profile_budget().__dict__.copy(),
        "governance_role": "qa",
        "governance_role_trusted": True,
    }
    qa_controller = GovernanceController(qa_agent, qa_controller.config)
    qa_controller.begin_turn(user_message="qa child", system_message="system", messages=[], task_id="child-qa")
    decision = qa_controller.before_model_call(request_id="child-qa:req:1", messages=[{"role": "user", "content": "qa probe"}], approx_request_tokens=50, api_call_count=0)
    assert decision["action"] in {"allow", "warning", "informational"}
    assert not decision.get("pause")
    assert not decision.get("stop")
    root_budget = json.loads((tmp_path / "engagements" / controller.engagement_id / "budget.json").read_text(encoding="utf-8"))
    assert root_budget["reservations"]["child-qa:req:1"]["agent_role"] == "qa"


def test_qa_stops_when_qa_reserve_is_exhausted(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="MiniCISO assessment", system_message="system", messages=[], task_id="root")
    budget = controller._profile_budget()
    assert controller._budget_file is not None
    controller._budget_file.update(
        lambda data: {
            **data,
            "consumed": {**data.get("consumed", {}), "total_tokens": budget.total_tokens - 20, "model_calls": budget.total_model_calls - 1},
        }
    )

    qa_controller, qa_agent = _make_controller(tmp_path, monkeypatch, mode="enforce", depth=1, role="qa", session_id="child-qa-stop")
    qa_agent._governance_engagement_id = controller.engagement_id
    qa_agent._governance_request_class = "engagement"
    qa_agent._governance_profile_key = "standard_engagement"
    qa_agent._governance_seed = {
        "engagement_id": controller.engagement_id,
        "request_class": "engagement",
        "profile_key": "standard_engagement",
        "budget_override": controller._profile_budget().__dict__.copy(),
        "governance_role": "qa",
        "governance_role_trusted": True,
    }
    qa_controller = GovernanceController(qa_agent, qa_controller.config)
    qa_controller.begin_turn(user_message="qa child", system_message="system", messages=[], task_id="child-qa-stop")
    decision = qa_controller.before_model_call(request_id="child-qa-stop:req:1", messages=[{"role": "user", "content": "qa probe"}], approx_request_tokens=50, api_call_count=0)
    assert decision["action"] == "hard_stop"
    assert decision["stop"] is True


def test_actor_cannot_self_declare_as_qa(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="MiniCISO assessment", system_message="system", messages=[], task_id="root")
    budget = controller._profile_budget()
    reserve = int(budget.total_tokens * budget.qa_reserve_ratio)
    assert controller._budget_file is not None
    controller._budget_file.update(
        lambda data: {
            **data,
            "consumed": {**data.get("consumed", {}), "total_tokens": budget.total_tokens - reserve + 10, "model_calls": 0},
        }
    )

    fake_qa_controller, fake_qa_agent = _make_controller(tmp_path, monkeypatch, mode="enforce", depth=1, role="qa", session_id="fake-qa")
    fake_qa_agent._governance_engagement_id = controller.engagement_id
    fake_qa_agent._governance_request_class = "engagement"
    fake_qa_agent._governance_profile_key = "standard_engagement"
    fake_qa_controller = GovernanceController(fake_qa_agent, fake_qa_controller.config)
    fake_qa_controller.begin_turn(user_message="fake qa", system_message="system", messages=[], task_id="fake-qa")
    decision = fake_qa_controller.before_model_call(request_id="fake-qa:req:1", messages=[{"role": "user", "content": "probe"}], approx_request_tokens=50, api_call_count=0)
    assert decision["action"] in {"approval", "hard_stop"}


def test_shared_root_envelope_is_safe_across_processes(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(
        user_message="MiniCISO assessment",
        system_message="system",
        messages=[],
        task_id="root-task",
    )
    config_mapping = {
        "mode": "enforce",
        "workspace_dir": str(tmp_path / "engagements"),
        "force_for_profiles": ["chief-of-staff"],
        "profiles": {
            "standard_engagement": {
                **controller._profile_budget().__dict__.copy(),
                "total_model_calls": 8,
                "total_tokens": 36000,
            }
        },
    }
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_process_child_consume,
            args=(config_mapping, "chief-of-staff", str(tmp_path), controller.engagement_id, f"child-{index}", "sme", queue),
        )
        for index in range(2)
    ]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join(30)
        assert proc.exitcode == 0
    snapshots = [queue.get(timeout=5) for _ in processes]
    assert snapshots[-1]["model_calls"] == 2
    root_budget = json.loads((tmp_path / "engagements" / controller.engagement_id / "budget.json").read_text(encoding="utf-8"))
    assert root_budget["consumed"]["model_calls"] == 2
    assert set(root_budget["children"]) >= {"child-0", "child-1"}


def test_process_safe_budget_locking(tmp_path):
    budget_path = tmp_path / "budget.json"
    budget = FileBackedBudget(budget_path)
    budget.initialize({
        "engagement_id": "eng-test", "budget": {}, "qa_reserve": {}, "consumed": {"model_calls": 0},
        "children": {}, "reservations": {}, "request_state": {}, "logical_calls": {},
        "budget_authorizations": {}, "accounting_latch": {},
    })

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    processes = [ctx.Process(target=_process_increment_budget, args=(str(budget_path), 25, queue)) for _ in range(4)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join(30)
        assert proc.exitcode == 0
    for _ in processes:
        assert queue.get(timeout=5) is True

    data = json.loads(budget_path.read_text(encoding="utf-8"))
    assert data["consumed"]["model_calls"] == 100
    assert not budget_path.with_suffix(".json.lock").exists()


def test_before_tool_call_detects_equivalent_calls_and_observe_mode_only_warns(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="observe")
    controller.begin_turn(user_message="review", system_message="system", messages=[], task_id="task-observe")

    first = controller.before_tool_call("read_file", {"path": "a.txt"})
    second = controller.before_tool_call("read_file", {"path": "a.txt"})
    third = controller.before_tool_call("read_file", {"path": "a.txt"})

    assert first["action"] == "allow"
    assert second["action"] in {"allow", "warning"}
    assert third["action"] == "warning"
    for decision in (first, second, third):
        assert decision["allowed"] is True
        assert isinstance(decision["reason"], str)
        assert decision["compact_context"] is False
        assert decision["termination_reason"] is None
    telemetry = (tmp_path / "engagements" / controller.engagement_id / "telemetry.jsonl").read_text(encoding="utf-8")
    assert "tool_threshold" in telemetry


def test_persist_partial_handoff_writes_resumable_artifact_and_resume_loads_state(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="observe")
    controller.begin_turn(
        user_message="MiniCISO assessment",
        system_message="system",
        messages=[],
        task_id="task-5",
    )
    brief_path = tmp_path / "engagements" / controller.engagement_id / "brief.json"
    brief = json.loads(brief_path.read_text(encoding="utf-8"))
    brief["scope"] = ["repo root"]
    brief["decisions"] = ["narrow toolset"]
    brief["evidence_ids"] = ["ev-1"]
    brief["claim_ids"] = ["cl-1"]
    brief["qa_obligations"] = ["review claim cl-1"]
    brief_path.write_text(json.dumps(brief), encoding="utf-8")

    payload = controller.persist_partial_handoff(
        task="QA review",
        status="timeout",
        errors=["timed out"],
        recommended_next_step="resume with narrower scope",
    )
    loaded = controller.load_partial_handoff(payload["resume_identifier"])

    assert payload["status"] == "timeout"
    assert payload["resume_token"].startswith(f"{controller.engagement_id}:task-5:")
    assert payload["artifact_references"]
    saved = controller.resolve_artifact_reference(payload["artifact_references"][0])
    assert not Path(payload["artifact_references"][0]).is_absolute()
    assert saved.exists()
    assert loaded["evidence_collected"] == ["ev-1"]
    assert loaded["claims_evaluated"] == ["cl-1"]
    assert loaded["preserved_brief"]["decisions"] == ["narrow toolset"]


def test_normalize_usage_payload_handles_supported_mock_shapes_and_fallback(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="observe")
    controller.begin_turn(user_message="assessment", system_message="system", messages=[], task_id="task-usage")

    openai_shape = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "prompt_tokens_details": {"cached_tokens": 25}}
    anthropic_shape = {"usage": {"input_tokens": 120, "output_tokens": 80, "cache_creation_tokens": 20}}
    malformed_shape = {"usage": {"prompt_tokens": "oops"}}

    normalized_openai = controller.normalize_usage_payload(openai_shape, approx_request_tokens=10, response_text="ok")
    normalized_anthropic = controller.normalize_usage_payload(anthropic_shape, approx_request_tokens=10, response_text="ok")
    normalized_bad = controller.normalize_usage_payload(malformed_shape, approx_request_tokens=200, response_text="final summary")

    assert normalized_openai["input_tokens"] == 100
    assert normalized_openai["cached_input_tokens"] == 25
    assert normalized_openai["cache_confirmed"] is True
    assert normalized_anthropic["input_tokens"] == 120
    assert normalized_anthropic["output_tokens"] == 80
    assert normalized_anthropic["cache_creation_tokens"] == 20
    assert normalized_bad["estimated"] is True
    assert normalized_bad["input_tokens"] >= 229
    assert normalized_bad["output_tokens"] >= 32


def test_record_compaction_preserves_scope_decisions_and_ids(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="observe")
    controller.begin_turn(user_message="MiniCISO assessment", system_message="system", messages=[], task_id="task-compact")
    brief_path = tmp_path / "engagements" / controller.engagement_id / "brief.json"
    brief = json.loads(brief_path.read_text(encoding="utf-8"))
    brief["scope"] = ["api"]
    brief["decisions"] = ["compress before overflow"]
    brief["evidence_ids"] = ["ev-123"]
    brief["claim_ids"] = ["cl-456"]
    brief["qa_obligations"] = ["validate claim cl-456"]
    brief_path.write_text(json.dumps(brief), encoding="utf-8")

    checkpoint = controller.record_compaction_event(
        reason="threshold_warning",
        pre_messages=[{"role": "user", "content": "a" * 4000}],
        post_messages=[{"role": "assistant", "content": "compacted"}],
        system_message="system",
    )
    payload = json.loads(Path(checkpoint).read_text(encoding="utf-8"))

    assert payload["label"] == "context-compaction"
    assert payload["preserved_brief"]["scope"] == ["api"]
    assert payload["preserved_brief"]["decisions"] == ["compress before overflow"]
    assert payload["preserved_brief"]["evidence_ids"] == ["ev-123"]
    assert payload["preserved_brief"]["claim_ids"] == ["cl-456"]
    assert payload["preserved_brief"]["qa_obligations"] == ["validate claim cl-456"]


def test_close_turn_reports_tool_overhead_and_usage(tmp_path, monkeypatch):
    controller, agent = _make_controller(tmp_path, monkeypatch, mode="observe")
    controller.begin_turn(
        user_message="MiniCISO assessment",
        system_message="system",
        messages=[],
        task_id="task-6",
    )
    agent.tools = [
        {"function": {"name": "read_file"}},
        {"function": {"name": "search_files"}},
    ]
    controller.record_tool_schema_metrics(agent.tools, [], total_available_count=5)
    controller._tools_used.add("read_file")
    request_id = _prepare_dispatched_request(controller, "task-6:req:1", estimate=80)
    controller.record_model_usage(None, request_id=request_id, duration_seconds=0.1, approx_request_tokens=80, response_text="ok")

    summary = controller.close_turn(status="completed", final_response="ok", termination_reason="completed")

    assert summary["status"] == "completed"
    assert summary["tool_overhead"]["definitions_sent"] == 2
    assert summary["tool_overhead"]["total_available"] == 5
    assert summary["tool_overhead"]["unused_tools"] == 1
    assert summary["usage"]["total_tokens"] > 0


def test_conversational_request_keeps_governance_overhead_minimal(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="observe")
    controller.begin_turn(user_message="oi, tudo bem?", system_message="system", messages=[], task_id="task-chat")
    request_id = _prepare_dispatched_request(controller, "task-chat:req:1", estimate=20)
    controller.record_model_usage(None, request_id=request_id, duration_seconds=0.05, approx_request_tokens=20, response_text="Tudo bem!")
    controller.close_turn(status="completed", final_response="Tudo bem!", termination_reason="completed")

    assert controller.request_class == "conversational"
    assert controller.profile_key == "conversational"
    assert controller.state is not None
    assert controller.state.qa_reserve["total_tokens"] == 0
    telemetry_lines = (tmp_path / "engagements" / controller.engagement_id / "telemetry.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(telemetry_lines) <= 6


@pytest.mark.parametrize("raw_id", [None, "None", "none", "NULL", "null", "", "   "])
def test_invalid_engagement_identity_is_replaced_before_path_construction(tmp_path, monkeypatch, raw_id):
    controller, agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    agent._governance_engagement_id = raw_id
    controller = GovernanceController(agent, controller.config)

    assert controller.engagement_id
    assert controller.engagement_id.lower() not in {"none", "null", "undefined"}
    assert controller._engagement_dir().name == controller.engagement_id
    assert controller._engagement_dir().resolve().is_relative_to(controller.root_dir.resolve())


def test_same_parent_and_child_share_engagement_identity(tmp_path, monkeypatch):
    parent, agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    parent.begin_turn(user_message="parent", system_message="system", messages=[], task_id="parent-task")

    child, child_agent = _make_controller(tmp_path, monkeypatch, mode="enforce", depth=1, role="sme", session_id="child-sme-shared")
    child_agent._governance_engagement_id = parent.engagement_id
    child_agent._governance_request_class = parent.request_class
    child_agent._governance_profile_key = parent.profile_key
    child_agent._governance_seed = {
        "engagement_id": parent.engagement_id,
        "request_class": parent.request_class,
        "profile_key": parent.profile_key,
    }
    child = GovernanceController(child_agent, parent.config)

    assert child.engagement_id == parent.engagement_id
    assert child._engagement_dir() == parent._engagement_dir()


def test_same_engagement_uses_same_path_across_instances(tmp_path, monkeypatch):
    controller, agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    agent._governance_engagement_id = "eng-fixed"
    one = GovernanceController(agent, controller.config)
    two = GovernanceController(agent, controller.config)

    assert one.engagement_id == two.engagement_id == "eng-fixed"
    assert one._engagement_dir() == two._engagement_dir()


def test_reconcile_uses_persisted_reservation_estimate_when_usage_missing(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="MiniCISO", system_message="system", messages=[], task_id="acct-missing")
    request_id = _prepare_dispatched_request(controller, "acct:req:1", estimate=60)

    canonical = controller.record_model_usage(None, request_id=request_id, duration_seconds=0.1, response_text="")
    budget = json.loads((tmp_path / "engagements" / controller.engagement_id / "budget.json").read_text(encoding="utf-8"))

    assert canonical["estimated"] is True
    assert budget["consumed"]["total_tokens"] >= 60
    assert budget["reservations"] == {}
    assert budget["request_state"][request_id]["state"] == "completed"


def test_reconcile_is_idempotent_for_same_request_id(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="MiniCISO", system_message="system", messages=[], task_id="acct-idem")
    request_id = _prepare_dispatched_request(controller, "acct:req:idem", estimate=40)

    first = controller.reconcile_usage(request_id, {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}, duration_seconds=0.1, response_text="ok")
    second = controller.reconcile_usage(request_id, {"prompt_tokens": 999, "completion_tokens": 999, "total_tokens": 1998}, duration_seconds=0.1, response_text="ok")
    budget = json.loads((tmp_path / "engagements" / controller.engagement_id / "budget.json").read_text(encoding="utf-8"))

    assert first["total_tokens"] == 30
    assert second["total_tokens"] == 30
    assert budget["consumed"]["total_tokens"] == 30
    assert budget["reservations"] == {}
    assert budget["request_state"][request_id]["state"] == "reconciled"


def test_account_dispatched_provider_error_preserves_conservative_consumption(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="MiniCISO", system_message="system", messages=[], task_id="acct-provider-error")
    request_id = _prepare_dispatched_request(controller, "acct:req:provider", estimate=55)

    controller.account_dispatched_request_failure(
        request_id,
        reason="provider_error",
        duration_seconds=0.2,
        retry_count=1,
        response_text="",
        error_message="provider boom",
    )
    budget = json.loads((tmp_path / "engagements" / controller.engagement_id / "budget.json").read_text(encoding="utf-8"))

    assert budget["consumed"]["total_tokens"] >= 55
    assert budget["reservations"] == {}
    assert budget["request_state"][request_id]["state"] == "provider_error_accounted"
    assert budget["request_state"][request_id]["termination_reason"] == "provider_error"


def test_record_model_usage_accounting_error_fail_closed_in_enforce(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="MiniCISO", system_message="system", messages=[], task_id="acct-fail-closed")
    request_id = _prepare_dispatched_request(controller, "acct:req:accterr", estimate=45)

    original = controller.reconcile_usage

    def boom(*args, **kwargs):
        raise RuntimeError("reconcile exploded")

    controller.reconcile_usage = boom
    canonical = controller.record_model_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, request_id=request_id, duration_seconds=0.1, response_text="ok")
    controller.reconcile_usage = original
    budget = json.loads((tmp_path / "engagements" / controller.engagement_id / "budget.json").read_text(encoding="utf-8"))

    assert canonical["total_tokens"] >= 15
    assert budget["consumed"]["total_tokens"] >= 15
    assert budget["reservations"] == {}
    assert budget["request_state"][request_id]["state"] == "accounting_error"

    blocked = controller.before_model_call(
        request_id="acct:req:next",
        messages=[{"role": "user", "content": "continue"}],
        approx_request_tokens=10,
        api_call_count=1,
    )
    assert blocked["action"] == "hard_stop"


def test_malformed_usage_falls_back_to_persisted_estimate(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="MiniCISO", system_message="system", messages=[], task_id="acct-malformed")
    request_id = _prepare_dispatched_request(controller, "acct:req:badusage", estimate=70)

    canonical = controller.record_model_usage({"usage": {"total_tokens": "nope"}}, request_id=request_id, duration_seconds=0.1, response_text="partial")
    budget = json.loads((tmp_path / "engagements" / controller.engagement_id / "budget.json").read_text(encoding="utf-8"))

    assert canonical["estimated"] is True
    assert budget["consumed"]["total_tokens"] >= 70
    assert budget["request_state"][request_id]["state"] == "completed"
    assert budget["reservations"] == {}


def test_unknown_request_id_reconciliation_raises(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="MiniCISO", system_message="system", messages=[], task_id="acct-unknown")

    with pytest.raises(RuntimeError, match="Invalid request state transition for missing:req: created -> reconciled"):
        controller.reconcile_usage("missing:req", {"total_tokens": 10}, duration_seconds=0.1, response_text="")


def test_request_state_machine_contract_and_invalid_transitions(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    assert controller._request_transition_map()["reconciled"] == {"completed"}
    assert "reconciled" not in controller._request_terminal_states()
    assert "completed" in controller._request_terminal_states()

    controller.begin_turn(user_message="MiniCISO", system_message="system", messages=[], task_id="state-machine")
    request_id = _prepare_dispatched_request(controller, "acct:req:state", estimate=30)
    controller.reconcile_usage(request_id, {"total_tokens": 30}, duration_seconds=0.1, response_text="ok")
    controller.finalize_request(request_id, "success", "completed")

    budget = json.loads((tmp_path / "engagements" / controller.engagement_id / "budget.json").read_text(encoding="utf-8"))
    assert budget["request_state"][request_id]["state"] == "completed"

    with pytest.raises(RuntimeError, match="Cannot reserve request from state completed"):
        controller.reserve_request(request_id, 30)
    with pytest.raises(RuntimeError, match="Cannot dispatch request without reservation"):
        controller.record_dispatch(request_id)
    repeated = controller.reconcile_usage(request_id, {"total_tokens": 30}, duration_seconds=0.1, response_text="ok")
    assert repeated["total_tokens"] == 30


def test_accounting_latch_persists_across_controller_reconstruction(tmp_path, monkeypatch):
    controller, agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="MiniCISO", system_message="system", messages=[], task_id="latch-root")
    request_id = _prepare_dispatched_request(controller, "acct:req:latch", estimate=45)
    controller.account_dispatched_request_failure(
        request_id,
        reason="accounting_error",
        duration_seconds=0.1,
        retry_count=0,
        response_text="",
        error_message="boom",
    )

    snapshot = controller._root_snapshot()
    assert snapshot["accounting_latch"]["active"] is True
    assert request_id in snapshot["accounting_latch"]["request_ids"]

    agent._governance_engagement_id = controller.engagement_id
    agent._governance_request_class = controller.request_class
    agent._governance_profile_key = controller.profile_key
    agent._governance_seed = {
        "engagement_id": controller.engagement_id,
        "request_class": controller.request_class,
        "profile_key": controller.profile_key,
    }
    rebuilt = GovernanceController(agent, controller.config)
    rebuilt.begin_turn(user_message="MiniCISO", system_message="system", messages=[], task_id="latch-root")
    blocked = rebuilt.before_model_call(
        request_id="acct:req:new",
        messages=[{"role": "user", "content": "continue"}],
        approx_request_tokens=10,
        api_call_count=1,
    )

    assert blocked["action"] == "hard_stop"
    rebuilt_snapshot = rebuilt._root_snapshot()
    assert rebuilt_snapshot["accounting_latch"]["active"] is True
    assert request_id in rebuilt_snapshot["accounting_latch"]["request_ids"]


def test_allocate_request_identity_persists_attempt_ordinals_across_reconstruction(tmp_path, monkeypatch):
    controller, agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="MiniCISO", system_message="system", messages=[], task_id="identity-root")

    first = controller.allocate_request_identity(logical_call_id="turn-1:logical:1")
    second = controller.allocate_request_identity(logical_call_id="turn-1:logical:1")

    agent._governance_engagement_id = controller.engagement_id
    agent._governance_request_class = controller.request_class
    agent._governance_profile_key = controller.profile_key
    agent._governance_seed = {
        "engagement_id": controller.engagement_id,
        "request_class": controller.request_class,
        "profile_key": controller.profile_key,
    }
    rebuilt = GovernanceController(agent, controller.config)
    rebuilt.begin_turn(user_message="MiniCISO", system_message="system", messages=[], task_id="identity-root")
    third = rebuilt.allocate_request_identity(logical_call_id="turn-1:logical:1")

    assert first["attempt_ordinal"] == 1
    assert second["attempt_ordinal"] == 2
    assert third["attempt_ordinal"] == 3
    assert first["attempt_id"] == "turn-1:logical:1:attempt:1"
    assert second["attempt_id"] == "turn-1:logical:1:attempt:2"
    assert third["attempt_id"] == "turn-1:logical:1:attempt:3"
    assert len({first["request_id"], second["request_id"], third["request_id"]}) == 3

    budget = json.loads((tmp_path / "engagements" / controller.engagement_id / "budget.json").read_text(encoding="utf-8"))
    logical_entry = budget["logical_calls"]["turn-1:logical:1"]
    assert logical_entry["last_allocated_attempt_ordinal"] == 3
    assert logical_entry["next_attempt_ordinal"] == 4


def test_informational_decision_is_explicitly_allowed_in_observe(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="observe")
    controller.begin_turn(user_message="review", system_message="system", messages=[], task_id="info-observe")
    monkeypatch.setattr(controller, "_threshold_action", lambda _ratio: "informational")

    decision = controller.before_model_call(
        request_id="info-observe:req:1",
        logical_call_id="info-observe:logical:1",
        messages=[{"role": "user", "content": "review"}],
        approx_request_tokens=100,
        api_call_count=0,
    )

    assert decision["allowed"] is True
    assert decision["action"] == "informational"
    assert decision["reason"] == "threshold_informational_observed"
    assert decision["compact_context"] is False
    assert decision["termination_reason"] is None
    assert decision["reservation_id"] == "info-observe:req:1"


def test_observe_never_blocks_even_when_budget_admission_is_exceeded(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="observe")
    controller.begin_turn(user_message="review", system_message="system", messages=[], task_id="observe-over")
    budget = controller._profile_budget()
    assert controller._budget_file is not None
    controller._budget_file.update(
        lambda data: {
            **data,
            "consumed": {
                **data.get("consumed", {}),
                "total_tokens": budget.total_tokens + 1,
                "model_calls": budget.total_model_calls + 1,
            },
        }
    )
    monkeypatch.setattr(controller, "_threshold_action", lambda _ratio: "hard_stop")

    decision = controller.before_model_call(
        request_id="observe-over:req:1",
        logical_call_id="observe-over:logical:1",
        messages=[{"role": "user", "content": "continue"}],
        approx_request_tokens=100,
        api_call_count=100,
    )

    assert decision["allowed"] is True
    assert decision["action"] == "hard_stop"
    assert decision["compact_context"] is False
    assert decision["reservation_id"] == "observe-over:req:1"


def test_informational_action_can_be_explicitly_denied(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    decision = controller.validate_decision(
        {
            "allowed": False,
            "action": "informational",
            "reason": "owner_policy_denied",
            "compact_context": False,
            "termination_reason": "owner_policy_denied",
        },
        request_id="info-denied:req:1",
    )

    assert decision["allowed"] is False
    assert decision["action"] == "informational"


@pytest.mark.parametrize(
    "decision",
    [
        None,
        {},
        {"allowed": True, "action": "mystery", "reason": "x", "compact_context": False, "termination_reason": None},
        {"allowed": "yes", "action": "allow", "reason": "x", "compact_context": False, "termination_reason": None},
        {"allowed": True, "action": "allow", "reason": 42, "compact_context": False, "termination_reason": None},
    ],
)
def test_missing_unknown_or_malformed_decision_fails_closed_in_enforce(tmp_path, monkeypatch, decision):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    with pytest.raises(RuntimeError, match="invalid governance decision"):
        controller.validate_decision(decision, request_id="malformed:req:1")


@pytest.mark.parametrize("blocked_action", ["approval", "hard_stop"])
@pytest.mark.parametrize("envelope_operation", ["increment", "replace"])
def test_valid_human_authorization_applies_once_to_new_request_without_mutating_global_budget(tmp_path, monkeypatch, blocked_action, envelope_operation):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="security review", system_message="system", messages=[], task_id="auth-valid")
    monkeypatch.setattr(controller, "_threshold_action", lambda _ratio: blocked_action)
    budget_before = dict(controller._root_snapshot()["budget"])

    paused = controller.before_model_call(
        request_id="auth-valid:req:1",
        logical_call_id="auth-valid:logical:1",
        messages=[{"role": "user", "content": "continue"}],
        approx_request_tokens=700,
        api_call_count=0,
    )
    challenge = paused["authorization"]
    assert paused["allowed"] is False
    assert paused["pause"] is (blocked_action == "approval")
    assert paused["stop"] is (blocked_action == "hard_stop")
    assert "authenticated callback" in paused["message"]

    approved_envelope = _explicit_increment(tokens=1200, calls=1, per_request=800)
    approved_envelope["operation"] = envelope_operation
    approved = controller.submit_budget_authorization(
        authorization_id=challenge["authorization_id"],
        nonce=challenge["nonce"],
        callback_capability=controller.agent._governance_callback_capability,
        principal=_owner_principal(engagement_id=controller.engagement_id),
        approved=True,
        reason="authenticated owner decision",
        envelope=approved_envelope,
    )
    assert approved["status"] == "authorized"
    assert approved["authorized_by"]["subject"] == "owner-123"

    resumed = controller.before_model_call(
        request_id="auth-valid:req:2",
        logical_call_id="auth-valid:logical:1",
        messages=[{"role": "user", "content": "continue"}],
        approx_request_tokens=700,
        api_call_count=0,
    )
    assert resumed["allowed"] is True
    assert resumed["action"] == blocked_action
    assert resumed["reason"] == "human_budget_authorization"
    assert resumed["reservation_id"] == "auth-valid:req:2"
    snapshot = controller._root_snapshot()
    assert snapshot["budget"] == budget_before
    record = snapshot["budget_authorizations"][challenge["authorization_id"]]
    assert record["status"] == "consumed"
    assert record["consumed_by_request_id"] == "auth-valid:req:2"
    assert record["consumed_by_logical_call_id"] == "auth-valid:logical:1"
    assert snapshot["consumed"]["total_tokens"] == 0
    budget_text = (tmp_path / "engagements" / controller.engagement_id / "budget.json").read_text(encoding="utf-8")
    assert challenge["nonce"] not in budget_text
    audit_path = tmp_path / "engagements" / controller.engagement_id / "authorization_audit.jsonl"
    audit_events = [json.loads(line)["event"] for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert audit_events == ["authorization_requested", "authorization_authorized", "authorization_consumed"]

    with pytest.raises(RuntimeError, match="already consumed"):
        controller.submit_budget_authorization(
            authorization_id=challenge["authorization_id"],
            nonce=challenge["nonce"],
        callback_capability=controller.agent._governance_callback_capability,
            principal=_owner_principal(engagement_id=controller.engagement_id),
            approved=True,
            reason="authenticated owner decision",
            envelope=_explicit_increment(),
        )


def test_expired_authorization_fails_closed(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="security review", system_message="system", messages=[], task_id="auth-expired")
    challenge = controller.create_budget_authorization(
        blocked_request_id="auth-expired:req:1",
        logical_call_id="auth-expired:logical:1",
        decision_action="approval",
        proposed_envelope=_explicit_increment(),
        ttl_seconds=1,
    )
    controller._budget_file.update(
        lambda data: {
            **data,
            "budget_authorizations": {
                **data["budget_authorizations"],
                challenge["authorization_id"]: {
                    **data["budget_authorizations"][challenge["authorization_id"]],
                    "expires_at": "2000-01-01T00:00:00+00:00",
                },
            },
        }
    )

    with pytest.raises(RuntimeError, match="expired"):
        controller.submit_budget_authorization(
            authorization_id=challenge["authorization_id"],
            nonce=challenge["nonce"],
        callback_capability=controller.agent._governance_callback_capability,
            principal=_owner_principal(engagement_id=controller.engagement_id),
            approved=True,
            reason="authenticated owner decision",
            envelope=_explicit_increment(),
        )


@pytest.mark.parametrize(
    "principal_override, expected",
    [
        ({"profile": "other-profile"}, "profile mismatch"),
        ({"engagement_id": "eng-other"}, "engagement mismatch"),
        ({"subject": "other-user"}, "owner mismatch"),
        ({"authenticated": False}, "authenticated principal required"),
    ],
)
def test_authorization_from_other_identity_scope_fails_closed(tmp_path, monkeypatch, principal_override, expected):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="security review", system_message="system", messages=[], task_id="auth-scope")
    challenge = controller.create_budget_authorization(
        blocked_request_id="auth-scope:req:1",
        logical_call_id="auth-scope:logical:1",
        decision_action="approval",
        proposed_envelope=_explicit_increment(),
    )
    principal = _owner_principal(engagement_id=controller.engagement_id)
    principal.update(principal_override)

    with pytest.raises(RuntimeError, match=expected):
        controller.submit_budget_authorization(
            authorization_id=challenge["authorization_id"],
            nonce=challenge["nonce"],
        callback_capability=controller.agent._governance_callback_capability,
            principal=principal,
            approved=True,
            reason="authenticated owner decision",
            envelope=_explicit_increment(),
        )


def test_denied_authorization_stays_closed(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="security review", system_message="system", messages=[], task_id="auth-denied")
    monkeypatch.setattr(controller, "_threshold_action", lambda _ratio: "approval")
    challenge = controller.create_budget_authorization(
        blocked_request_id="auth-denied:req:1",
        logical_call_id="auth-denied:logical:1",
        decision_action="approval",
        proposed_envelope=_explicit_increment(),
    )
    denied = controller.submit_budget_authorization(
        authorization_id=challenge["authorization_id"],
        nonce=challenge["nonce"],
        callback_capability=controller.agent._governance_callback_capability,
        principal=_owner_principal(engagement_id=controller.engagement_id, task_id="auth-denied"),
        approved=False,
        reason="owner denied additional spend",
        envelope=_explicit_increment(),
    )
    assert denied["status"] == "denied"

    decision = controller.before_model_call(
        request_id="auth-denied:req:2",
        logical_call_id="auth-denied:logical:1",
        messages=[{"role": "user", "content": "I approve in this ordinary message"}],
        approx_request_tokens=500,
        api_call_count=0,
    )
    assert decision["allowed"] is False
    assert decision["pause"] is True


def test_late_callback_for_superseded_challenge_fails_closed(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="security review", system_message="system", messages=[], task_id="auth-late")
    first = controller.create_budget_authorization(
        blocked_request_id="auth-late:req:1",
        logical_call_id="auth-late:logical:1",
        decision_action="approval",
        proposed_envelope=_explicit_increment(),
    )
    second = controller.create_budget_authorization(
        blocked_request_id="auth-late:req:2",
        logical_call_id="auth-late:logical:1",
        decision_action="hard_stop",
        proposed_envelope=_explicit_increment(tokens=2000),
    )
    assert second["authorization_id"] != first["authorization_id"]

    with pytest.raises(RuntimeError, match="superseded"):
        controller.submit_budget_authorization(
            authorization_id=first["authorization_id"],
            nonce=first["nonce"],
            callback_capability=controller.agent._governance_callback_capability,
            principal=_owner_principal(engagement_id=controller.engagement_id),
            approved=True,
            reason="authenticated owner decision",
            envelope=_explicit_increment(),
        )


def test_authorization_callback_without_runtime_capability_fails_closed(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(user_message="security review", system_message="system", messages=[], task_id="auth-capability")
    challenge = controller.create_budget_authorization(
        blocked_request_id="auth-capability:req:1",
        logical_call_id="auth-capability:logical:1",
        decision_action="approval",
        proposed_envelope=_explicit_increment(),
    )

    with pytest.raises(RuntimeError, match="authenticated callback capability required"):
        controller.submit_budget_authorization(
            authorization_id=challenge["authorization_id"],
            nonce=challenge["nonce"],
            callback_capability=None,
            principal=_owner_principal(engagement_id=controller.engagement_id),
            approved=True,
            reason="authenticated owner decision",
            envelope=_explicit_increment(),
        )
