import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import CompressionGovernanceStop
from agent.cost_context_governance import GovernanceController
from agent.tool_executor import execute_tool_calls_sequential
from run_agent import AIAgent


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _base_governance_config(tmp_path: Path) -> dict:
    return {
        "cost_context_governance": {
            "mode": "enforce",
            "workspace_dir": str(tmp_path / "engagements"),
            "force_for_profiles": ["chief-of-staff"],
            "role_toolsets": {
                "chief": ["file"],
                "sme": ["file"],
                "qa": ["file"],
            },
            "threshold": {
                "informational_ratio": 0.3,
                "warning_ratio": 0.4,
                "approval_ratio": 0.8,
                "hard_stop_ratio": 1.0,
            },
            "progress": {
                "equivalent_tool_calls": 2,
                "same_error_repeats": 2,
                "no_progress_iterations": 2,
            },
            "profiles": {
                "conversational": {
                    "total_model_calls": 3,
                    "calls_per_agent": 3,
                    "calls_per_delegated_task": 0,
                    "input_tokens": 2000,
                    "output_tokens": 1000,
                    "total_tokens": 3000,
                    "context_tokens_per_request": 400,
                    "wall_clock_seconds": 60,
                    "tool_invocations": 4,
                    "retries": 1,
                    "delegation_count": 0,
                    "iterations_without_progress": 2,
                    "qa_reserve_ratio": 0.0,
                },
                "bounded": {
                    "total_model_calls": 2,
                    "calls_per_agent": 2,
                    "calls_per_delegated_task": 1,
                    "input_tokens": 3000,
                    "output_tokens": 1500,
                    "total_tokens": 4500,
                    "context_tokens_per_request": 300,
                    "wall_clock_seconds": 60,
                    "tool_invocations": 3,
                    "retries": 1,
                    "delegation_count": 1,
                    "iterations_without_progress": 2,
                    "qa_reserve_ratio": 0.1,
                },
                "standard_engagement": {
                    "total_model_calls": 3,
                    "calls_per_agent": 2,
                    "calls_per_delegated_task": 1,
                    "input_tokens": 6000,
                    "output_tokens": 3000,
                    "total_tokens": 9000,
                    "context_tokens_per_request": 5000,
                    "wall_clock_seconds": 60,
                    "tool_invocations": 4,
                    "retries": 1,
                    "delegation_count": 1,
                    "iterations_without_progress": 2,
                    "qa_reserve_ratio": 0.2,
                },
            },
        }
    }


def _merge(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def _mock_response(content="ok", *, finish_reason="stop", usage=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    response = SimpleNamespace(choices=[choice], model="test/model")
    response.usage = SimpleNamespace(**usage) if usage else None
    return response


def _tool_call(call_id: str, name: str = "read_file", arguments: str = '{"path": "a.txt"}'):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def _make_agent(tmp_path: Path, monkeypatch, *, config_override: dict | None = None) -> AIAgent:
    config = _base_governance_config(tmp_path)
    if config_override:
        config = _merge(config, config_override)

    monkeypatch.setenv("HERMES_PROFILE", "chief-of-staff")
    monkeypatch.setenv("HERMES_CONFIG_PATH", str(tmp_path / "profiles" / "chief-of-staff" / "config.yaml"))
    (tmp_path / "profiles" / "chief-of-staff").mkdir(parents=True, exist_ok=True)

    def _defs(enabled_toolsets=None, **_kwargs):
        if enabled_toolsets == ["file"]:
            return _tool_defs("read_file", "search_files")
        return _tool_defs("read_file", "search_files", "browser_navigate", "terminal")

    with (
        patch("run_agent.get_tool_definitions", side_effect=_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=config),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
        patch("run_agent._hermes_home", tmp_path),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "system"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._persist_session = MagicMock()
    agent._save_trajectory = MagicMock()
    agent._cleanup_task_resources = MagicMock()
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    agent._append_guardrail_observation = MagicMock(side_effect=lambda _name, _args, result, **_kwargs: result)
    agent._record_file_mutation_result = MagicMock()
    agent._subdirectory_hints.check_tool_call = MagicMock(return_value="")
    agent._tool_result_content_for_active_model = MagicMock(side_effect=lambda _name, result: result)
    agent.context_compressor.should_compress = lambda *_args, **_kwargs: False
    return agent


@pytest.fixture()
def agent(tmp_path, monkeypatch):
    return _make_agent(tmp_path, monkeypatch)


def test_tool_schema_filtering_reaches_provider_payload(agent, monkeypatch):
    monkeypatch.setattr(agent.cost_context_governance, "_threshold_action", lambda _ratio: "allow")
    captured = {}
    response = _mock_response(
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )

    def _capture(**kwargs):
        captured.update(kwargs)
        return response

    agent.client.chat.completions.create.side_effect = _capture
    result = agent.run_conversation("Preciso de uma análise MiniCISO de segurança")

    assert result["completed"] is True
    assert {tool["function"]["name"] for tool in agent.tools} == {"read_file", "search_files"}
    assert {tool["function"]["name"] for tool in captured["tools"]} == {"read_file", "search_files"}
    assert "browser_navigate" not in json.dumps(captured["tools"])
    assert agent.cost_context_governance._tool_schema_metrics["definitions_sent"] == 2
    assert agent.cost_context_governance._tool_schema_metrics["serialized_schema_size"] > 0


def test_schema_filter_failure_never_sends_unfiltered_tools(agent, monkeypatch):
    canary = (
        "secret-schema-filter path=/home/vpsadmin/.hermes/private/schema.json"
        " url=https://example.invalid/filter?token=do-not-leak"
    )
    with patch.object(
        GovernanceController,
        "filter_tool_schemas",
        side_effect=RuntimeError(canary),
    ):
        rebuilt = _make_agent(agent.cost_context_governance.root_dir.parent.parent, monkeypatch)
    monkeypatch.setattr(rebuilt.cost_context_governance, "_threshold_action", lambda _ratio: "allow")

    captured = {}
    rebuilt.client.chat.completions.create.side_effect = lambda **kwargs: (
        captured.update(kwargs) or _mock_response(
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
    )
    result = rebuilt.run_conversation("security review")

    assert rebuilt.tools == []
    assert not captured.get("tools")
    assert result["completed"] is True
    assert canary not in json.dumps(result)
    for path in Path(rebuilt.cost_context_governance.root_dir).rglob("*"):
        if path.is_file():
            assert canary not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("mode, expect_provider", [("enforce", False), ("observe", True)])
def test_request_identity_failure_preserves_enforce_and_observe(mode, expect_provider, tmp_path, monkeypatch):
    canary = (
        "secret-identity path=/home/vpsadmin/.hermes/private/identity.json"
        " url=https://example.invalid/identity?token=do-not-leak"
    )
    rebuilt = _make_agent(
        tmp_path,
        monkeypatch,
        config_override={"cost_context_governance": {"mode": mode}},
    )
    monkeypatch.setattr(rebuilt.cost_context_governance, "_threshold_action", lambda _ratio: "allow")
    provider_calls = []
    rebuilt.client.chat.completions.create.side_effect = lambda **_kwargs: (
        provider_calls.append(True)
        or _mock_response(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    )
    monkeypatch.setattr(
        rebuilt.cost_context_governance,
        "allocate_request_identity",
        MagicMock(side_effect=RuntimeError(canary)),
    )

    result = rebuilt.run_conversation("security review")

    assert bool(provider_calls) is expect_provider
    if mode == "enforce":
        assert result["completed"] is False
        assert result["failed"] is True
    else:
        assert result["completed"] is True
    assert canary not in json.dumps(result)
    assert not any(path.name == "None" for path in Path(tmp_path / "engagements").glob("*"))
    for path in Path(rebuilt.cost_context_governance.root_dir).rglob("*"):
        if path.is_file():
            assert canary not in path.read_text(encoding="utf-8")


def test_logical_compaction_result_failure_stops_before_provider(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    canary = (
        "secret-compaction path=/home/vpsadmin/.hermes/private/compaction.json"
        " url=https://example.invalid/compact?token=do-not-leak"
    )
    decisions = iter([
        {
            "allowed": False,
            "action": "warning",
            "reason": "threshold_warning_requires_compaction",
            "compact_context": True,
            "termination_reason": "threshold_warning",
        },
        {
            "allowed": True,
            "action": "allow",
            "reason": "must_not_reach_provider",
            "compact_context": False,
            "termination_reason": None,
        },
    ])
    monkeypatch.setattr(
        agent.cost_context_governance,
        "before_model_call",
        lambda **kwargs: {**next(decisions), "request_id": kwargs["request_id"]},
    )
    agent._compress_context = MagicMock(
        return_value=([{"role": "user", "content": "compacted"}], "system")
    )
    record_result = MagicMock(side_effect=RuntimeError(canary))
    monkeypatch.setattr(
        agent.cost_context_governance,
        "record_logical_compaction_result",
        record_result,
    )
    provider_calls = []
    agent.client.chat.completions.create.side_effect = lambda **_kwargs: (
        provider_calls.append(True)
        or _mock_response(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    )

    result = agent.run_conversation("security review")

    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "governance_logical_compaction_persistence_error"
    assert record_result.call_count == 1
    assert not provider_calls
    assert canary not in json.dumps(result)
    for path in Path(agent.cost_context_governance.root_dir).rglob("*"):
        if path.is_file():
            assert canary not in path.read_text(encoding="utf-8")


def test_governance_compaction_stop_does_not_expose_raw_exception(agent, monkeypatch):
    canary = (
        "secret-canary-7f4d"
        " path=/home/vpsadmin/.hermes/private/token.json"
        " url=https://example.invalid/callback?access_token=do-not-leak"
    )
    decision = {
        "allowed": True,
        "compact_context": True,
        "pause": False,
        "stop": False,
    }
    monkeypatch.setattr(
        agent.cost_context_governance,
        "before_model_call",
        lambda **_kwargs: decision,
    )
    monkeypatch.setattr(
        agent.cost_context_governance,
        "validate_decision",
        lambda value, **_kwargs: value,
    )
    agent._compress_context = MagicMock(
        side_effect=CompressionGovernanceStop(canary),
    )
    agent.client.chat.completions.create.side_effect = AssertionError(
        "provider must remain blocked"
    )

    result = agent.run_conversation("compact safely")

    assert canary not in result["final_response"]
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "cost_context_governance_hard_stop"
    assert result["final_response"] == (
        "Execução pausada pela governança de custo/contexto durante a compactação."
    )
    assert agent.client.chat.completions.create.call_count == 0
    for path in Path(agent.cost_context_governance.root_dir).rglob("*"):
        if path.is_file():
            assert canary not in path.read_text(encoding="utf-8")


def test_before_model_call_allow_path_precedes_provider_and_reconciles_usage(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(agent.cost_context_governance, "_threshold_action", lambda _ratio: "allow")
    response = _mock_response(content="ok", usage={"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100})
    agent.client.chat.completions.create.return_value = response
    trail = []
    original = agent.cost_context_governance.before_model_call

    def wrapped_before_model_call(**kwargs):
        trail.append("before_model_call")
        return original(**kwargs)

    def provider(**kwargs):
        trail.append("provider")
        return response

    agent.cost_context_governance.before_model_call = wrapped_before_model_call
    agent.client.chat.completions.create.side_effect = provider

    result = agent.run_conversation("Preciso de uma análise MiniCISO de segurança")

    assert result["completed"] is True
    assert trail == ["before_model_call", "provider"]
    assert agent.client.chat.completions.create.call_count == 1
    candidates = [p for p in (Path(tmp_path) / "engagements").glob("*") if (p / "budget.json").exists() and p.name != "None"]
    assert candidates
    budget_path = candidates[-1] / "budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    assert budget["consumed"]["total_tokens"] == 100
    assert budget.get("reservations", {}) == {}
    assert budget["reservation_reconciliation"]["reserved_tokens"] >= 1
    assert budget["reservation_reconciliation"]["actual_total_tokens"] == 100


def test_before_model_call_blocks_provider_and_persists_partial_handoff(tmp_path, monkeypatch):
    agent = _make_agent(
        tmp_path,
        monkeypatch,
        config_override={
            "cost_context_governance": {
                "profiles": {
                    "standard_engagement": {
                        "total_tokens": 700,
                        "input_tokens": 500,
                        "output_tokens": 200,
                        "context_tokens_per_request": 250,
                    }
                }
            }
        },
    )
    trail = []
    original = agent.cost_context_governance.before_model_call
    agent.client.chat.completions.create.side_effect = AssertionError("provider must not be called")

    def wrapped_before_model_call(**kwargs):
        trail.append("before_model_call")
        return original(**kwargs)

    agent.cost_context_governance.before_model_call = wrapped_before_model_call
    result = agent.run_conversation("Preciso de uma análise MiniCISO de segurança " + ("x " * 2000))

    assert trail == ["before_model_call"]
    assert agent.client.chat.completions.create.call_count == 0
    assert result["completed"] is True
    assert "Hard stop" in result["final_response"]
    candidates = [p for p in (Path(tmp_path) / "engagements").glob("*") if (p / "summary.json").exists()]
    assert candidates
    engagement_dir = candidates[-1]
    summary = json.loads((engagement_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["termination_reason"] == "cost_context_governance_hard_stop"
    handoffs = sorted((engagement_dir / "partial_handoffs").glob("*.json"))
    assert handoffs
    payload = json.loads(handoffs[-1].read_text(encoding="utf-8"))
    assert payload["status"] == "hard_stop"
    assert payload["termination_reason"] == "hard_stop"
    assert payload["resume_identifier"]
    assert payload["artifact_references"]
    assert payload["usage"]["model_calls"] == 0


def test_compaction_happens_before_provider_and_provider_receives_compacted_context(tmp_path, monkeypatch):
    agent = _make_agent(
        tmp_path,
        monkeypatch,
        config_override={
            "cost_context_governance": {
                "enabled": True,
                "model_profiles": {
                    "default": "conversational",
                    "by_provider": {"": "conversational"},
                    "budgets": {
                        "conversational": {
                            "context_tokens_per_request": 60000,
                            "total_tokens": 200000,
                            "input_tokens": 150000,
                            "output_tokens": 50000,
                            "total_model_calls": 12,
                        }
                    },
                },
            }
        },
    )
    response = _mock_response(content="ok", usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
    agent.compression_enabled = True
    agent.context_compressor.threshold_tokens = 1000
    agent.context_compressor.tail_token_budget = 600
    agent.context_compressor._augment_summary_lean = lambda summary, _turns: summary
    agent.client.chat.completions.create.return_value = response
    trail = []
    calls = {"count": 0}
    seen_request_ids = []
    released = []
    aux_calls = []
    original_before_model_call = agent.cost_context_governance.before_model_call
    original_release_request = agent.cost_context_governance.release_request

    def before_model_call(**kwargs):
        calls["count"] += 1
        seen_request_ids.append(kwargs.get("request_id"))
        request_kind = kwargs.get("request_kind") or "primary"
        trail.append(f"before_model_call:{calls['count']}:{request_kind}")
        if calls["count"] == 1 and request_kind == "primary":
            return {
                "request_id": kwargs["request_id"],
                "allowed": False,
                "action": "warning",
                "reason": "threshold_warning_requires_compaction",
                "compact_context": True,
                "termination_reason": "threshold_warning",
            }
        if request_kind == "compaction":
            kwargs = dict(kwargs)
            kwargs["approx_request_tokens"] = min(int(kwargs.get("approx_request_tokens", 0) or 0), 1000)
            kwargs["messages"] = [{"role": "user", "content": "compaction preflight seam"}]
            return original_before_model_call(**kwargs)
        elif request_kind == "primary" and calls["count"] >= 3:
            kwargs = dict(kwargs)
            payload = agent.cost_context_governance.evaluate_request(
                request_id=kwargs["request_id"],
                messages=[{"role": "user", "content": "primary post-compaction seam"}],
                approx_request_tokens=1000,
                current_model_calls=int(kwargs.get("api_call_count", 0) or 0),
                logical_call_id=kwargs.get("logical_call_id"),
                attempt_id=kwargs.get("attempt_id"),
                request_kind="primary",
            )
            reservation = agent.cost_context_governance.reserve_request(kwargs["request_id"], 1000)
            return {
                **payload,
                "allowed": True,
                "action": "allow",
                "reason": "test_post_compaction_allowed",
                "compact_context": False,
                "termination_reason": None,
                "reservation_id": reservation.get("reservation_id"),
            }

    def fake_aux_llm(**kwargs):
        aux_calls.append(kwargs)
        trail.append("aux_provider")
        return _mock_response(
            content="COMPACTED scope=repo evidence=ev-1 claim=cl-1 qa=qa-1 artifact=artifact-1",
            usage={"prompt_tokens": 123, "completion_tokens": 40, "total_tokens": 163},
        )

    def provider(**kwargs):
        trail.append("provider")
        return response

    def release_request(request_id, reason):
        released.append((request_id, reason))
        return original_release_request(request_id, reason)

    agent.cost_context_governance.before_model_call = before_model_call
    agent.cost_context_governance.release_request = release_request
    agent.client.chat.completions.create.side_effect = provider
    agent.cost_context_governance.begin_turn(
        user_message="MiniCISO security review",
        system_message="system",
        messages=[{"role": "user", "content": "seed"}],
        task_id="compaction-check",
    )
    brief_path = Path(agent.cost_context_governance.root_dir) / agent.cost_context_governance.engagement_id / "brief.json"
    brief = json.loads(brief_path.read_text(encoding="utf-8"))
    brief.update(
        {
            "scope": ["repo:miniCISO"],
            "evidence_ids": ["ev-1"],
            "claim_ids": ["cl-1"],
            "qa_obligations": ["qa-1"],
            "artifact_references": ["artifact-1"],
        }
    )
    brief_path.write_text(json.dumps(brief), encoding="utf-8")
    conversation_history = [
        {"role": "user", "content": "Contexto prévio de segurança " + ("a " * 4000)},
        {"role": "assistant", "content": "Resposta prévia de assessment " + ("b " * 4000)},
        {"role": "user", "content": "Mais evidências coletadas " + ("c " * 4000)},
        {"role": "assistant", "content": "Mais hipóteses e impacto " + ("d " * 4000)},
        {"role": "user", "content": "Achados adicionais para consolidar " + ("e " * 4000)},
        {"role": "assistant", "content": "Rascunho intermediário do relatório " + ("f " * 4000)},
        {"role": "user", "content": "Nova trilha de exploração " + ("g " * 4000)},
        {"role": "assistant", "content": "Nova análise parcial " + ("h " * 4000)},
    ]

    with agent.context_compressor.bind_auxiliary_runtime(
        aux_llm_callable=fake_aux_llm,
        aux_llm_configured=True,
    ):
        result = agent.run_conversation(
            "Preciso de uma análise MiniCISO de segurança " + ("x " * 1500),
            conversation_history=conversation_history,
        )

    assert result["completed"] is True
    assert trail[0] == "before_model_call:1:primary"
    assert "aux_provider" in trail
    assert "provider" in trail
    assert trail.index("aux_provider") < trail.index("provider")
    assert len([rid for rid in seen_request_ids if rid]) >= 3
    assert len(set(rid for rid in seen_request_ids if rid)) >= 3
    assert released == [(seen_request_ids[0], "governance_warning_compaction")]
    sent_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert len(sent_messages) < len(conversation_history) + 2
    assert any(
        message.get("role") == "assistant"
        and "COMPACTED scope=repo evidence=ev-1 claim=cl-1 qa=qa-1 artifact=artifact-1" in str(message.get("content") or "")
        for message in sent_messages
    )
    assert aux_calls
    assert list(aux_calls[0].get("messages") or [])
    checkpoint_dir = Path(agent.cost_context_governance.root_dir) / agent.cost_context_governance.engagement_id / "checkpoints"
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in checkpoint_dir.glob("*.json")]
    compaction = next(item for item in payloads if item.get("label") == "context-compaction")
    assert compaction["preserved_brief"]["scope"] == ["repo:miniCISO"]
    assert compaction["preserved_brief"]["evidence_ids"] == ["ev-1"]
    assert compaction["preserved_brief"]["claim_ids"] == ["cl-1"]
    assert compaction["preserved_brief"]["qa_obligations"] == ["qa-1"]
    assert compaction["preserved_brief"]["artifact_references"] == ["artifact-1"]


def test_tool_executor_blocking_prevents_tool_body_execution(agent):
    agent.cost_context_governance.begin_turn(
        user_message="review",
        system_message="system",
        messages=[],
        task_id="tool-runtime",
    )
    assistant = SimpleNamespace(tool_calls=[_tool_call("1"), _tool_call("2")])
    messages: list[dict] = []
    executed = []

    def fake_tool(_name, _args, _task_id, *, tool_call_id, **_kwargs):
        executed.append(tool_call_id)
        return "tool-result"

    decisions = iter([
        {
            "allowed": True,
            "action": "allow",
            "reason": "test_allowed",
            "compact_context": False,
            "termination_reason": None,
        },
        {
            "allowed": False,
            "action": "hard_stop",
            "reason": "tool_threshold_hard_stop",
            "compact_context": False,
            "termination_reason": "cost_context_governance_tool_hard_stop",
            "message": "Governança bloqueou nova chamada para read_file.",
        },
    ])
    agent.cost_context_governance.before_tool_call = lambda *_args, **_kwargs: next(decisions)
    with patch("run_agent.handle_function_call", side_effect=fake_tool):
        execute_tool_calls_sequential(agent, assistant, messages, "tool-runtime")

    assert executed == ["1"]
    assert len(messages) == 2
    assert "Governança bloqueou" in messages[1]["content"]


def test_accounting_latch_blocks_provider_but_persists_handoff(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(agent.cost_context_governance, "_threshold_action", lambda _ratio: "allow")
    response = _mock_response(content="ok", usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
    provider_calls = {"count": 0}
    original_reconcile = agent.cost_context_governance.reconcile_usage

    def provider(**_kwargs):
        provider_calls["count"] += 1
        return response

    def exploding_reconcile(*args, **kwargs):
        if provider_calls["count"] == 1:
            raise RuntimeError("synthetic reconcile failure")
        return original_reconcile(*args, **kwargs)

    agent.client.chat.completions.create.side_effect = provider
    agent.cost_context_governance.reconcile_usage = exploding_reconcile
    first = agent.run_conversation("Preciso de uma análise MiniCISO de segurança")
    assert first["completed"] is True
    assert provider_calls["count"] == 1

    second = agent.run_conversation("Continue a análise MiniCISO")
    assert second["completed"] is True
    assert provider_calls["count"] == 1
    assert "Governança bloqueou novas chamadas" in second["final_response"]

    candidates = [p for p in (Path(tmp_path) / "engagements").glob("*") if (p / "budget.json").exists()]
    assert candidates
    engagement_dir = candidates[-1]
    budget = json.loads((engagement_dir / "budget.json").read_text(encoding="utf-8"))
    assert budget["accounting_latch"]["active"] is True
    handoffs = sorted((engagement_dir / "partial_handoffs").glob("*.json"))
    checkpoints = sorted((engagement_dir / "checkpoints").glob("*.json"))
    assert handoffs
    assert checkpoints
    latest_handoff = json.loads(handoffs[-1].read_text(encoding="utf-8"))
    assert latest_handoff["status"] == "hard_stop"
    assert latest_handoff["resume_identifier"]
    labels = [json.loads(path.read_text(encoding="utf-8"))["label"] for path in checkpoints]
    assert "hard-stop-accounting-error" in labels


def test_observe_mode_records_same_telemetry_without_blocking(tmp_path, monkeypatch):
    observe_agent = _make_agent(tmp_path, monkeypatch, config_override={"cost_context_governance": {"mode": "observe"}})
    observe_agent.cost_context_governance.begin_turn(
        user_message="MiniCISO assessment",
        system_message="system",
        messages=[],
        task_id="observe-runtime",
    )

    first = observe_agent.cost_context_governance.before_tool_call("read_file", {"path": "a.txt"})
    second = observe_agent.cost_context_governance.before_tool_call("read_file", {"path": "a.txt"})

    assert first["action"] == "allow"
    assert second["action"] in {"allow", "warning"}
    telemetry_path = Path(observe_agent.cost_context_governance.root_dir) / observe_agent.cost_context_governance.engagement_id / "telemetry.jsonl"
    telemetry = telemetry_path.read_text(encoding="utf-8")
    assert "tool_threshold" in telemetry


@pytest.mark.timeout(5)
def test_warning_compaction_no_progress_hard_stops_before_provider(tmp_path, monkeypatch):
    agent = _make_agent(
        tmp_path,
        monkeypatch,
        config_override={
            "cost_context_governance": {
                "threshold": {
                    "warning_ratio": 0.7,
                    "approval_ratio": 0.95,
                    "hard_stop_ratio": 1.0,
                },
                "profiles": {
                    "conversational": {
                        "context_tokens_per_request": 1800,
                        "total_tokens": 12000,
                        "input_tokens": 8000,
                        "output_tokens": 4000,
                    }
                }
            }
        },
    )
    agent.compression_enabled = True
    agent.client.chat.completions.create.side_effect = AssertionError("provider must not be called")

    seen_request_ids = []
    original = agent.cost_context_governance.before_model_call
    calls = {"count": 0, "forced": False}

    def before_model_call(**kwargs):
        calls["count"] += 1
        seen_request_ids.append(kwargs.get("request_id"))
        decision = original(**kwargs)
        if (
            not calls["forced"]
            and kwargs.get("request_kind") == "primary"
            and len(list(kwargs.get("messages") or [])) >= 3
        ):
            calls["forced"] = True
            return {
                "request_id": kwargs["request_id"],
                "allowed": False,
                "action": "warning",
                "reason": "threshold_warning_requires_compaction",
                "compact_context": True,
                "termination_reason": "threshold_warning",
            }
        return decision

    def compress(messages, system_message, **_kwargs):
        agent.context_compressor._last_compression_telemetry = None
        return list(messages), system_message

    agent.cost_context_governance.before_model_call = before_model_call
    agent._compress_context = compress
    conversation_history = [
        {"role": "user", "content": "Contexto prévio de segurança " + ("a " * 600)},
        {"role": "assistant", "content": "Resposta prévia de assessment " + ("b " * 600)},
    ]

    result = agent.run_conversation(
        "Preciso de uma análise MiniCISO de segurança " + ("x " * 1500),
        conversation_history=conversation_history,
    )

    assert result["completed"] is True
    assert "logical_call_no_progress" in result["final_response"]
    assert agent.client.chat.completions.create.call_count == 0
    assert len(seen_request_ids) == 1

    candidates = [p for p in (Path(tmp_path) / "engagements").glob("*") if (p / "budget.json").exists()]
    assert candidates
    budget = json.loads((candidates[-1] / "budget.json").read_text(encoding="utf-8"))
    assert budget.get("reservations", {}) == {}
    assert {
        state["state"] for state in budget.get("request_state", {}).values()
    } <= {
        "completed",
        "released_before_dispatch",
        "provider_error_accounted",
        "interrupted_accounted",
        "accounting_error",
        "blocked",
    }
    logical_calls = budget.get("logical_calls", {})
    assert logical_calls
    entry = next(iter(logical_calls.values()))
    assert entry["termination_reason"] == "logical_call_no_progress"
    assert entry["compaction_attempts"] == 1


def test_compaction_aux_model_call_gets_own_governed_request(tmp_path, monkeypatch):
    agent = _make_agent(
        tmp_path,
        monkeypatch,
        config_override={
            "cost_context_governance": {
                "threshold": {
                    "warning_ratio": 0.7,
                    "approval_ratio": 0.95,
                    "hard_stop_ratio": 1.0,
                },
                "profiles": {
                    "conversational": {
                        "context_tokens_per_request": 5000,
                        "total_tokens": 12000,
                        "input_tokens": 8000,
                        "output_tokens": 4000,
                    }
                }
            }
        },
    )
    agent.compression_enabled = True
    agent.context_compressor.threshold_tokens = 1000
    agent.context_compressor.tail_token_budget = 600
    response = _mock_response(content="ok", usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
    agent.client.chat.completions.create.return_value = response
    calls = {"count": 0, "forced": False}
    original = agent.cost_context_governance.before_model_call

    def before_model_call(**kwargs):
        calls["count"] += 1
        decision = original(**kwargs)
        if (
            not calls["forced"]
            and kwargs.get("request_kind") == "primary"
            and len(list(kwargs.get("messages") or [])) >= 3
        ):
            calls["forced"] = True
            return {
                "request_id": kwargs["request_id"],
                "allowed": False,
                "action": "warning",
                "reason": "threshold_warning_requires_compaction",
                "compact_context": True,
                "termination_reason": "threshold_warning",
            }
        return decision

    aux_response = _mock_response(
        content="COMPACTED scope=repo evidence=ev-1 claim=cl-1 qa=qa-1 artifact=artifact-1",
        usage={"prompt_tokens": 123, "completion_tokens": 40, "total_tokens": 163},
    )
    aux_calls = []

    def fake_aux_llm(**kwargs):
        aux_calls.append(kwargs)
        return aux_response

    monkeypatch.setattr(agent.context_compressor, "_augment_summary_lean", lambda summary, _turns: summary)
    agent.cost_context_governance.before_model_call = before_model_call
    conversation_history = [
        {"role": "user", "content": "Contexto prévio de segurança " + ("a " * 400)},
        {"role": "assistant", "content": "Resposta prévia de assessment " + ("b " * 400)},
        {"role": "user", "content": "Mais evidências coletadas " + ("c " * 400)},
        {"role": "assistant", "content": "Mais hipóteses e impacto " + ("d " * 400)},
        {"role": "user", "content": "Achados adicionais para consolidar " + ("e " * 400)},
        {"role": "assistant", "content": "Rascunho intermediário do relatório " + ("f " * 400)},
        {"role": "user", "content": "Nova trilha de exploração " + ("g " * 400)},
        {"role": "assistant", "content": "Nova análise parcial " + ("h " * 400)},
    ]

    with agent.context_compressor.bind_auxiliary_runtime(
        aux_llm_callable=fake_aux_llm,
        aux_llm_configured=True,
    ):
        result = agent.run_conversation(
            "Preciso de uma análise MiniCISO de segurança " + ("x " * 1500),
            conversation_history=conversation_history,
        )

    assert aux_calls
    assert list(aux_calls[0].get("messages") or [])
    assert result["completed"] is True
    candidates = [p for p in (Path(tmp_path) / "engagements").glob("*") if (p / "budget.json").exists()]
    assert candidates
    budget = json.loads((candidates[-1] / "budget.json").read_text(encoding="utf-8"))
    compaction_requests = {
        request_id: state
        for request_id, state in budget.get("request_state", {}).items()
        if ":compaction:" in request_id
    }
    assert compaction_requests
    compaction_state = next(iter(compaction_requests.values()))
    assert compaction_state["state"] == "completed"
    assert compaction_state["request_kind"] == "compaction"
    canonical_usage = compaction_state["canonical_usage"]
    assert canonical_usage["input_tokens"] == 123
    assert canonical_usage["output_tokens"] == 40
    assert canonical_usage["total_tokens"] == 163
    assert canonical_usage["estimated"] is False
    reconciliation = compaction_state["reconciliation"]
    assert reconciliation["actual_model_calls"] == 1
    assert reconciliation["actual_total_tokens"] == 163
    assert compaction_state["dispatch_count"] == 1


@pytest.mark.parametrize(
    "bad_decision",
    [
        None,
        {"allowed": True, "action": "unknown", "reason": "x", "compact_context": False, "termination_reason": None},
        {"allowed": "yes", "action": "allow", "reason": "x", "compact_context": False, "termination_reason": None},
        {"allowed": True, "action": "allow", "reason": None, "compact_context": "no", "termination_reason": None},
    ],
)
def test_enforce_invalid_decision_blocks_before_provider(tmp_path, monkeypatch, bad_decision):
    agent = _make_agent(tmp_path, monkeypatch)
    agent.client.chat.completions.create.side_effect = AssertionError("provider must not be called")
    original = agent.cost_context_governance.before_model_call
    monkeypatch.setattr(agent.cost_context_governance, "_threshold_action", lambda _ratio: "allow")

    def invalid_after_real_preflight(**kwargs):
        original(**kwargs)
        return bad_decision

    agent.cost_context_governance.before_model_call = invalid_after_real_preflight
    result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert "invalid governance decision" in result["final_response"]
    assert agent.client.chat.completions.create.call_count == 0


def test_enforce_informational_but_denied_blocks_before_provider(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    agent.client.chat.completions.create.side_effect = AssertionError("provider must not be called")
    original = agent.cost_context_governance.before_model_call
    monkeypatch.setattr(agent.cost_context_governance, "_threshold_action", lambda _ratio: "allow")

    def denied_after_real_preflight(**kwargs):
        original(**kwargs)
        return {
            "allowed": False,
            "action": "informational",
            "reason": "owner_policy_denied",
            "compact_context": False,
            "termination_reason": "owner_policy_denied",
            "request_id": kwargs["request_id"],
        }

    agent.cost_context_governance.before_model_call = denied_after_real_preflight
    result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert "owner_policy_denied" in result["final_response"]
    assert agent.client.chat.completions.create.call_count == 0


def test_observe_invalid_decision_never_blocks_provider_or_dispatch(tmp_path, monkeypatch):
    agent = _make_agent(
        tmp_path,
        monkeypatch,
        config_override={"cost_context_governance": {"mode": "observe"}},
    )
    response = _mock_response(
        content="provider-ok",
        usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    )
    agent.client.chat.completions.create.return_value = response
    original = agent.cost_context_governance.before_model_call
    monkeypatch.setattr(agent.cost_context_governance, "_threshold_action", lambda _ratio: "allow")

    def malformed_after_real_preflight(**kwargs):
        original(**kwargs)
        return None

    agent.cost_context_governance.before_model_call = malformed_after_real_preflight
    result = agent.run_conversation("hello")

    assert result["final_response"] == "provider-ok"
    assert agent.client.chat.completions.create.call_count == 1


def test_authenticated_authorization_resumes_as_new_request_and_accounts_once(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    controller = agent.cost_context_governance
    agent.platform = "telegram"
    agent._user_id = "owner-123"
    agent._chat_id = "chat-123"
    agent._governance_owner_identity_trusted = True
    agent._governance_callback_capability = object()
    monkeypatch.setattr(controller, "_threshold_action", lambda _ratio: "approval")
    response = _mock_response(
        content="resumed",
        usage={"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
    )
    agent.client.chat.completions.create.return_value = response
    captured = {}
    original = controller.before_model_call

    def capture_challenge(**kwargs):
        decision = original(**kwargs)
        if decision.get("authorization"):
            captured["decision"] = decision
        return decision

    controller.before_model_call = capture_challenge
    first = agent.run_conversation("pause for authenticated approval")

    assert first["completed"] is True
    assert agent.client.chat.completions.create.call_count == 0
    challenge = captured["decision"]["authorization"]
    controller.submit_budget_authorization(
        authorization_id=challenge["authorization_id"],
        nonce=challenge["nonce"],
        callback_capability=agent._governance_callback_capability,
        principal={
            "authenticated": True,
            "authentication_source": "gateway_adapter",
            "subject": "owner-123",
            "platform": "telegram",
            "profile": controller.profile_name,
            "engagement_id": controller.engagement_id,
            "chat_id": "chat-123",
            "task_id": controller.task_id,
        },
        approved=True,
        envelope={
            "operation": "increment",
            "total_tokens": 2_000,
            "total_model_calls": 1,
            "context_tokens_per_request": 2_000,
        },
        reason="runtime owner approval",
    )

    controller.before_model_call = original
    second = agent.run_conversation("resume through authenticated callback")

    assert second["completed"] is True
    assert second["final_response"] == "resumed"
    assert agent.client.chat.completions.create.call_count == 1
    budget = json.loads(controller._budget_file.path.read_text(encoding="utf-8"))
    requests = budget["request_state"]
    blocked = [request_id for request_id, state in requests.items() if state["state"] == "blocked"]
    completed = [request_id for request_id, state in requests.items() if state["state"] == "completed"]
    assert len(blocked) == 1
    assert len(completed) == 1
    assert blocked[0] != completed[0]
    assert requests[blocked[0]]["logical_call_id"] == requests[completed[0]]["logical_call_id"]
    assert requests[blocked[0]]["attempt_id"] != requests[completed[0]]["attempt_id"]
    authorization = budget["budget_authorizations"][challenge["authorization_id"]]
    assert authorization["status"] == "consumed"
    assert authorization["consumed_by_request_id"] == completed[0]
    assert budget["consumed"]["total_tokens"] == 100
    assert budget["consumed"]["model_calls"] == 1
    assert budget.get("reservations", {}) == {}


def test_preflight_compression_path_initializes_effective_task_id_before_use(
    tmp_path, monkeypatch
):
    agent = _make_agent(tmp_path, monkeypatch)
    agent.cost_context_governance = None
    agent.compression_enabled = True
    agent.context_compressor.protect_first_n = 0
    agent.context_compressor.protect_last_n = 0
    agent.context_compressor.should_compress = lambda _tokens: True
    agent._compress_context = lambda messages, system_message, **kwargs: (
        messages,
        system_message,
    )
    agent.client.chat.completions.create.return_value = _mock_response(content="ok")

    result = agent.run_conversation(
        "hello",
        conversation_history=[{"role": "user", "content": "previous"}],
    )

    assert result["final_response"] == "ok"


def test_dispatch_persistence_failure_blocks_provider_and_accounts_reservation(
    tmp_path, monkeypatch
):
    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(agent.cost_context_governance, "_threshold_action", lambda _ratio: "allow")
    agent.cost_context_governance.record_dispatch = MagicMock(
        side_effect=RuntimeError("dispatch persistence unavailable")
    )
    agent.client.chat.completions.create.return_value = _mock_response(content="must not run")

    result = agent.run_conversation("security review")

    assert agent.client.chat.completions.create.call_count == 0
    assert agent._append_guardrail_observation.call_count == 0
    assert result["completed"] is False
    assert result["failed"] is True
    budget = json.loads(agent.cost_context_governance._budget_file.path.read_text())
    assert budget["reservations"] == {}
    assert any(state["state"] == "accounting_error" for state in budget["request_state"].values())


def test_governance_artifacts_store_metadata_not_raw_canaries(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    controller = agent.cost_context_governance
    canaries = {
        "prompt": "PROMPT-CANARY-private-document",
        "args": "ARGS-CANARY-secret-argument",
        "result": "RESULT-CANARY-sensitive-output",
        "exception": "EXCEPTION-CANARY-/absolute/private/path",
        "url": "URL-CANARY?token=private-query",
        "header": "HEADER-CANARY-Bearer private-secret",
    }
    controller.begin_turn(
        user_message=canaries["prompt"],
        system_message=canaries["prompt"],
        messages=[{"role": "user", "content": canaries["prompt"]}],
        task_id="artifact-task",
    )
    controller.persist_checkpoint(
        "canary",
        {
            "messages_tail": [{"role": "user", "content": canaries["prompt"]}],
            "tool_args": {"secret": canaries["args"], "url": canaries["url"]},
            "result_preview": canaries["result"],
            "authorization_header": canaries["header"],
        },
    )
    controller.persist_partial_handoff(
        task="artifact-task",
        status="error",
        summary=canaries["result"],
        errors=[canaries["exception"]],
        recommended_next_step=canaries["url"],
    )
    controller.log_event("exception", {"error": canaries["exception"], "url": canaries["url"]})

    files = list((tmp_path / "engagements").rglob("*"))
    raw = b"".join(path.read_bytes() for path in files if path.is_file())
    for canary in canaries.values():
        assert canary.encode() not in raw
    brief = json.loads(next(p for p in files if p.name == "brief.json").read_text())
    assert brief["engagement_id"] == controller.engagement_id
    assert brief["resource_envelope"]
    assert controller._context_manifest


def test_dispatch_failure_public_error_is_stable_and_secondary_persistence_fails_closed(
    tmp_path, monkeypatch
):
    agent = _make_agent(tmp_path, monkeypatch)
    controller = agent.cost_context_governance
    controller.account_dispatched_request_failure = MagicMock(
        side_effect=RuntimeError("/absolute/private/path SECRET-CANARY")
    )
    controller.persist_checkpoint = MagicMock(side_effect=OSError("checkpoint unavailable"))
    controller.persist_partial_handoff = MagicMock(side_effect=OSError("handoff unavailable"))

    with pytest.raises(RuntimeError) as raised:
        controller.handle_dispatch_persistence_failure(
            "request-1", RuntimeError("/absolute/private/path SECRET-CANARY")
        )

    assert str(raised.value) == "Governança não confirmou o dispatch persistido; provider bloqueado."
    assert controller._dispatch_failure_latched is True
