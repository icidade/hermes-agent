import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.cost_context_governance import GovernanceConfig, GovernanceController
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


def _agent(tmp_path, *, user_id="owner-123", chat_id="chat-123", task_id="task-1"):
    agent = SimpleNamespace(
        _delegate_depth=0,
        _governance_engagement_id=None,
        _governance_request_class=None,
        _governance_profile_key=None,
        _governance_seed=None,
        _governance_role="chief",
        _governance_owner_identity_trusted=True,
        _governance_callback_capability=object(),
        _user_id=user_id,
        _chat_id=chat_id,
        platform="telegram",
        profile="chief-of-staff",
        session_id=task_id,
        provider="test",
        model="test-model",
        tools=[],
        client=MagicMock(),
    )
    cfg = GovernanceConfig.from_mapping({
        "mode": "enforce",
        "workspace_dir": str(tmp_path / "engagements"),
        "force_for_profiles": ["chief-of-staff"],
    })
    controller = GovernanceController(agent, cfg)
    controller.begin_turn(
        user_message="security review",
        system_message="system",
        messages=[],
        task_id=task_id,
    )
    agent.cost_context_governance = controller
    return agent, controller


def _envelope():
    return {
        "operation": "increment",
        "total_tokens": 1000,
        "total_model_calls": 1,
        "context_tokens_per_request": 800,
    }


def _challenge(controller, logical_call_id="logical-1"):
    return controller.create_budget_authorization(
        blocked_request_id=f"{logical_call_id}:blocked",
        logical_call_id=logical_call_id,
        decision_action="approval",
        proposed_envelope=_envelope(),
    )


def _principal(controller):
    return {
        "authenticated": True,
        "authentication_source": "gateway_adapter",
        "subject": "owner-123",
        "platform": "telegram",
        "profile": controller.profile_name,
        "engagement_id": controller.engagement_id,
        "chat_id": "chat-123",
        "task_id": controller.task_id,
    }


def _telegram_callback(agent, runner, token, *, user_id="owner-123", chat_id="chat-123", decision="a"):
    answered = []

    async def answer(**kwargs):
        answered.append(kwargs)

    query = SimpleNamespace(
        data=f"ba:{token}:{decision}",
        from_user=SimpleNamespace(id=user_id),
        message=SimpleNamespace(
            chat=SimpleNamespace(id=chat_id, type="private"),
            message_thread_id=None,
        ),
        answer=answer,
    )
    adapter = object.__new__(TelegramAdapter)
    adapter._budget_authorization_handler = runner._make_budget_authorization_handler()
    adapter._is_callback_user_authorized = lambda *_args, **_kwargs: True
    adapter._bot = object()
    asyncio.run(adapter._handle_budget_authorization_callback(query))
    assert answered
    return answered[-1]["text"]


def _runner_for(agent):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {"telegram-key": agent}
    runner._session_key_for_source = lambda _source: "telegram-key"
    return runner


def test_real_telegram_gateway_controller_approval_uses_opaque_single_use_callback_and_no_llm(tmp_path, monkeypatch):
    agent, controller = _agent(tmp_path)
    challenge = _challenge(controller)
    token = challenge["callback_token"]

    assert token
    assert len(f"ba:{token}:a".encode()) <= 64
    assert token not in (tmp_path / "engagements" / controller.engagement_id / "budget.json").read_text()
    assert "nonce" not in challenge["instructions"].lower()

    runner = _runner_for(agent)
    text = _telegram_callback(agent, runner, token)

    assert "processada" in text.lower()
    assert agent.client.chat.completions.create.call_count == 0
    record = controller._root_snapshot()["budget_authorizations"][challenge["authorization_id"]]
    assert record["status"] == "authorized"

    with pytest.raises(RuntimeError):
        controller.submit_budget_authorization_callback(
            callback_token=token,
            principal={
                "authenticated": True,
                "authentication_source": "gateway_adapter",
                "subject": "owner-123",
                "platform": "telegram",
                "chat_id": "chat-123",
            },
            approved=True,
            reason="replay",
        )


def test_real_telegram_gateway_controller_denial_never_calls_llm(tmp_path):
    agent, controller = _agent(tmp_path)
    challenge = _challenge(controller)
    text = _telegram_callback(agent, _runner_for(agent), challenge["callback_token"], decision="d")

    assert "processada" in text.lower()
    assert agent.client.chat.completions.create.call_count == 0
    assert controller._root_snapshot()["budget_authorizations"][challenge["authorization_id"]]["status"] == "denied"


def test_callback_resume_keeps_logical_call_but_allocates_new_attempt_and_request(tmp_path, monkeypatch):
    agent, controller = _agent(tmp_path)
    monkeypatch.setattr(controller, "_threshold_action", lambda _ratio: "approval")
    paused = controller.before_model_call(
        request_id="logical-1:blocked-request",
        logical_call_id="logical-1",
        messages=[{"role": "user", "content": "pause"}],
        approx_request_tokens=100,
        api_call_count=0,
        attempt_id="attempt-1",
    )
    challenge = paused["authorization"]
    _telegram_callback(agent, _runner_for(agent), challenge["callback_token"])

    resumed = controller.before_model_call(
        request_id="logical-1:resumed-request",
        logical_call_id="logical-1",
        messages=[{"role": "user", "content": "resume"}],
        approx_request_tokens=100,
        api_call_count=0,
        attempt_id="attempt-2",
    )

    assert resumed["allowed"] is True
    state = controller._root_snapshot()["request_state"]
    assert state["logical-1:blocked-request"]["logical_call_id"] == "logical-1"
    assert state["logical-1:resumed-request"]["logical_call_id"] == "logical-1"
    assert state["logical-1:blocked-request"]["attempt_id"] != state["logical-1:resumed-request"]["attempt_id"]
    assert "logical-1:blocked-request" != "logical-1:resumed-request"
    assert agent.client.chat.completions.create.call_count == 0


def test_authorization_cannot_be_reused_for_another_logical_call(tmp_path, monkeypatch):
    agent, controller = _agent(tmp_path)
    monkeypatch.setattr(controller, "_threshold_action", lambda _ratio: "approval")
    challenge = _challenge(controller, logical_call_id="logical-1")
    _telegram_callback(agent, _runner_for(agent), challenge["callback_token"])

    decision = controller.before_model_call(
        request_id="logical-2:new-request",
        logical_call_id="logical-2",
        messages=[{"role": "user", "content": "different operation"}],
        approx_request_tokens=100,
        api_call_count=0,
        attempt_id="attempt-other-logical-call",
    )

    assert decision["allowed"] is False
    assert controller._root_snapshot()["budget_authorizations"][challenge["authorization_id"]]["status"] == "authorized"
    assert agent.client.chat.completions.create.call_count == 0


@pytest.mark.parametrize(
    "field, value",
    [("subject", "other-user"), ("chat_id", "other-chat")],
)
def test_real_telegram_callback_rejects_other_user_or_chat(tmp_path, field, value):
    agent, controller = _agent(tmp_path)
    challenge = _challenge(controller)
    principal_user = value if field == "subject" else "owner-123"
    principal_chat = value if field == "chat_id" else "chat-123"

    text = _telegram_callback(
        agent,
        _runner_for(agent),
        challenge["callback_token"],
        user_id=principal_user,
        chat_id=principal_chat,
    )

    assert "denied" in text.lower() or "invalid" in text.lower()
    assert agent.client.chat.completions.create.call_count == 0
    assert controller._root_snapshot()["budget_authorizations"][challenge["authorization_id"]]["status"] == "pending"


def test_callback_expiry_replay_and_concurrent_callbacks_are_single_use(tmp_path):
    agent, controller = _agent(tmp_path)
    challenge = _challenge(controller)
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
        controller.submit_budget_authorization_callback(
            callback_token=challenge["callback_token"],
            principal=_principal(controller),
            approved=True,
            reason="expired",
        )

    agent, controller = _agent(tmp_path / "replay")
    challenge = _challenge(controller)
    outcomes = []
    barrier = threading.Barrier(2)

    def submit():
        barrier.wait()
        try:
            outcomes.append(controller.submit_budget_authorization_callback(
                callback_token=challenge["callback_token"],
                principal=_principal(controller),
                approved=True,
                reason="concurrent",
            ))
        except RuntimeError as exc:
            outcomes.append(exc)

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(isinstance(item, dict) for item in outcomes) == 1
    assert controller._root_snapshot()["budget_authorizations"][challenge["authorization_id"]]["status"] == "authorized"


def test_authorization_persistence_failure_never_looks_successful(tmp_path, monkeypatch):
    agent, controller = _agent(tmp_path)
    challenge = _challenge(controller)
    original = controller._budget_file.update

    def fail_once(mutator):
        monkeypatch.setattr(controller._budget_file, "update", original)
        raise OSError("storage unavailable")

    monkeypatch.setattr(controller._budget_file, "update", fail_once)
    with pytest.raises(OSError):
        controller.submit_budget_authorization_callback(
            callback_token=challenge["callback_token"],
            principal=_principal(controller),
            approved=True,
            reason="persist failure",
        )
    assert not hasattr(agent, "_governance_resume_authorization_id")


def test_consumption_persistence_failure_does_not_resume(tmp_path, monkeypatch):
    agent, controller = _agent(tmp_path)
    challenge = _challenge(controller)
    controller.submit_budget_authorization_callback(
        callback_token=challenge["callback_token"],
        principal=_principal(controller),
        approved=True,
        reason="owner approval",
    )
    original = controller._budget_file.update
    monkeypatch.setattr(controller, "_threshold_action", lambda _ratio: "approval")
    monkeypatch.setattr(controller._budget_file, "update", lambda _mutator: (_ for _ in ()).throw(OSError("storage unavailable")))
    with pytest.raises(OSError):
        controller.before_model_call(
            request_id="logical-1:new-request",
            logical_call_id="logical-1",
            messages=[{"role": "user", "content": "resume"}],
            approx_request_tokens=100,
            api_call_count=0,
        )
    monkeypatch.setattr(controller._budget_file, "update", original)
    assert getattr(agent, "_governance_resume_authorization_id", None)


def test_canonical_gateway_start_wires_real_telegram_adapter_to_profile_controller(
    tmp_path, monkeypatch
):
    """The production startup path installs the callback bridge on Telegram."""
    agent, controller = _agent(tmp_path)
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token"),
        },
        sessions_dir=tmp_path / "gateway-sessions",
    )
    runner = GatewayRunner(config)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-123",
        chat_type="dm",
        user_id="owner-123",
        profile=None,
    )
    session_key = runner._session_key_for_source(source)
    runner._agent_cache[session_key] = (agent, "test-signature")
    challenge = _challenge(controller, logical_call_id="canonical-logical")

    async def fake_connect(adapter, _platform):
        adapter._running = True
        return True

    monkeypatch.setattr(runner, "_connect_initial_adapter_with_timeout", fake_connect)
    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", lambda: _no_secondary())

    async def _no_secondary():
        return 0

    async def run_start_and_stop():
        started = await runner.start()
        first_adapter = runner.adapters[Platform.TELEGRAM]
        first_text = await _telegram_callback_async(first_adapter, challenge["callback_token"])
        restart_challenge = _challenge(controller, logical_call_id="canonical-restart-logical")
        restarted = await runner.start()
        second_adapter = runner.adapters[Platform.TELEGRAM]
        second_text = await _telegram_callback_async(
            second_adapter, restart_challenge["callback_token"]
        )
        await runner.stop()
        return (
            started,
            restarted,
            first_adapter,
            second_adapter,
            first_text,
            second_text,
            restart_challenge,
        )

    (
        started,
        restarted,
        adapter,
        restarted_adapter,
        text,
        restart_text,
        restart_challenge,
    ) = asyncio.run(run_start_and_stop())

    assert started is True
    assert restarted is True
    assert adapter.platform == Platform.TELEGRAM
    assert restarted_adapter.platform == Platform.TELEGRAM
    assert restarted_adapter is not adapter
    assert "processada" in text.lower()
    assert "processada" in restart_text.lower()
    record = controller._root_snapshot()["budget_authorizations"][challenge["authorization_id"]]
    assert record["status"] == "authorized"
    assert controller.profile_name == "default"
    assert record["authorized_by"]["profile"] == controller.profile_name
    assert record["authorized_by"]["subject"] == "owner-123"
    assert record["owner"]["chat_id"] == "chat-123"
    restart_record = controller._root_snapshot()["budget_authorizations"][restart_challenge["authorization_id"]]
    assert restart_record["status"] == "authorized"
    assert (tmp_path / "engagements" / controller.engagement_id / "budget.json").exists()
    assert agent.client.chat.completions.create.call_count == 0
    replay_text = asyncio.run(_telegram_callback_async(adapter, challenge["callback_token"]))
    assert "denied" in replay_text.lower() or "invalid" in replay_text.lower()
    assert agent.client.chat.completions.create.call_count == 0


async def _telegram_callback_async(adapter, token):
    answered = []

    async def answer(**kwargs):
        answered.append(kwargs)

    query = SimpleNamespace(
        data=f"ba:{token}:a",
        from_user=SimpleNamespace(id="owner-123"),
        message=SimpleNamespace(
            chat=SimpleNamespace(id="chat-123", type="private"),
            message_thread_id=None,
        ),
        answer=answer,
    )
    await adapter._handle_budget_authorization_callback(query)
    assert answered
    return answered[-1]["text"]


@pytest.mark.parametrize(
    ("chat_type", "thread_id", "expected_type", "expected_thread"),
    [
        ("private", None, "dm", None),
        ("group", None, "group", None),
        ("supergroup", None, "group", None),
        ("supergroup", "  ", "group", None),
        ("supergroup", "42", "forum", "42"),
    ],
)
def test_telegram_message_source_normalizes_chat_identity_once(
    chat_type, thread_id, expected_type, expected_thread
):
    adapter = object.__new__(TelegramAdapter)
    message = SimpleNamespace(
        from_user=SimpleNamespace(id="owner-123", username="owner"),
        chat=SimpleNamespace(id="chat-123", type=chat_type, is_forum=False),
        message_thread_id=thread_id,
        is_topic_message=thread_id is not None and bool(str(thread_id).strip()),
    )

    source = adapter._source_from_message_for_auth(message)

    assert source.chat_type == expected_type
    assert source.thread_id == expected_thread
