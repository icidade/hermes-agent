from types import SimpleNamespace

from agent import governance_admin
from gateway.run_inbound import GatewayInboundMixin


def test_governance_command_routes_before_agent_and_deduplicates_controller(monkeypatch):
    controller = object()
    agent = SimpleNamespace(cost_context_governance=controller)
    runner = GatewayInboundMixin.__new__(GatewayInboundMixin)
    runner._agent_cache = {"session": agent}
    runner._running_agents = {"session": agent}
    captured = {}

    def execute(text, *, source, controllers):
        captured.update(text=text, source=source, controllers=controllers)
        return "controlled"

    monkeypatch.setattr(governance_admin, "execute_gateway_command", execute)
    source = SimpleNamespace(platform="telegram", user_id="owner")
    event = SimpleNamespace(
        text="!governance status", internal=False, allow_gateway_control=True
    )

    assert runner._hm_handle_governance_admin(event, source) == "controlled"
    assert captured["controllers"] == [controller]


def test_untrusted_governance_text_does_not_enter_control_plane(monkeypatch):
    runner = GatewayInboundMixin.__new__(GatewayInboundMixin)
    runner._agent_cache = {}
    runner._running_agents = {}
    monkeypatch.setattr(
        governance_admin,
        "execute_gateway_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )
    source = SimpleNamespace(platform="slack", user_id="U1")
    event = SimpleNamespace(
        text="!governance mode log", internal=False, allow_gateway_control=False
    )

    assert runner._hm_handle_governance_admin(event, source) is None
