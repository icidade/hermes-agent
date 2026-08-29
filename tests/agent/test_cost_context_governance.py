import json
from pathlib import Path
from unittest.mock import MagicMock

from agent.cost_context_governance import (
    BudgetProfile,
    EnvelopeUsage,
    GovernanceController,
    GovernanceConfig,
)


def _make_agent(depth=0):
    agent = MagicMock()
    agent._delegate_depth = depth
    agent._governance_engagement_id = None
    agent._governance_request_class = None
    agent._governance_profile_key = None
    agent._governance_seed = None
    agent.provider = "openai-codex"
    agent.model = "gpt-5.4"
    agent.session_id = "sess-123"
    agent.tools = []
    agent.profile = "chief-of-staff"
    agent._governance_role = "chief" if depth == 0 else "sme"
    return agent


def _make_controller(tmp_path, monkeypatch, *, mode="observe", profile_name="chief-of-staff", depth=0):
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
    agent = _make_agent(depth=depth)
    controller = GovernanceController(agent, config)
    return controller, agent


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

    brief = Path(tmp_path / "engagements" / controller.engagement_id / "brief.json")
    assert brief.exists()
    data = json.loads(brief.read_text(encoding="utf-8"))
    assert data["request_class"] == "engagement"
    assert data["resource_envelope"]["total_model_calls"] == BudgetProfile().total_model_calls


def test_before_model_call_requests_pause_when_qa_reserve_would_be_violated(tmp_path, monkeypatch):
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
        messages=[{"role": "user", "content": "x" * 200}],
        approx_request_tokens=200,
        api_call_count=0,
    )

    assert decision["action"] == "approval"
    assert decision["pause"] is True


def test_before_tool_call_hard_stops_repeated_no_progress_in_enforce_mode(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="enforce")
    controller.begin_turn(
        user_message="review",
        system_message="system",
        messages=[],
        task_id="task-3",
    )
    assert controller.state is not None
    controller.state.usage.no_progress_iterations = controller._profile_budget().iterations_without_progress - 1

    decision = controller.before_tool_call("read_file", {"path": "/tmp/a.txt"})
    assert decision["action"] == "allow"

    blocked = controller.before_tool_call("read_file", {"path": "/tmp/a.txt"})
    assert blocked["action"] == "hard_stop"
    assert "Governança bloqueou" in blocked["message"]


def test_allocate_child_budget_respects_qa_headroom(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="observe")
    controller.begin_turn(
        user_message="MiniCISO assessment",
        system_message="system",
        messages=[],
        task_id="task-4",
    )
    budget = controller._profile_budget()
    reserve = int(budget.total_tokens * budget.qa_reserve_ratio)
    assert controller._budget_file is not None
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

    child = controller.allocate_child_budget(child_task_id="sa-1", requested_profile="standard_engagement")
    assert child["allowed"] is False
    assert child["allocated_budget"]["qa_reserve_ratio"] == 0.0


def test_allocate_child_budget_returns_seed_fields_for_child_handoff(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="observe")
    controller.begin_turn(
        user_message="MiniCISO assessment",
        system_message="system",
        messages=[],
        task_id="task-seed",
    )

    child = controller.allocate_child_budget(child_task_id="sa-seed", requested_profile="standard_engagement")

    assert child["engagement_id"] == controller.engagement_id
    assert child["request_class"] == "engagement"
    assert child["profile_key"] == "standard_engagement"
    assert child["budget_override"] == child["allocated_budget"]


def test_persist_partial_handoff_writes_resumable_artifact(tmp_path, monkeypatch):
    controller, _agent = _make_controller(tmp_path, monkeypatch, mode="observe")
    controller.begin_turn(
        user_message="MiniCISO assessment",
        system_message="system",
        messages=[],
        task_id="task-5",
    )

    payload = controller.persist_partial_handoff(
        task="QA review",
        status="timeout",
        errors=["timed out"],
        recommended_next_step="resume with narrower scope",
    )

    assert payload["status"] == "timeout"
    assert payload["resume_token"].startswith(f"{controller.engagement_id}:task-5:")
    assert payload["artifact_references"]
    saved = Path(payload["artifact_references"][0])
    assert saved.exists()


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
    controller._tools_used.add("read_file")
    assert controller.state is not None
    controller.state.usage = EnvelopeUsage(model_calls=1, total_tokens=123, tool_invocations=1)

    summary = controller.close_turn(status="completed", final_response="ok", termination_reason="completed")

    assert summary["status"] == "completed"
    assert summary["tool_overhead"]["definitions_sent"] == 2
    assert summary["tool_overhead"]["unused_tools"] == 1
    assert summary["usage"]["total_tokens"] == 123
