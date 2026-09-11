"""Regression coverage for Telegram scoped authorization failures."""

import logging
from types import SimpleNamespace

from agent import secret_scope
from gateway import authz_mixin
from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter, _scoped_gate_env


def _telegram_adapter() -> TelegramAdapter:
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter._message_handler = None
    return adapter


def _message_adapter(handler=None, *, extra=None) -> TelegramAdapter:
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="test", extra=dict(extra or {}))
    adapter._message_handler = handler
    adapter._authorization_check = None
    adapter._owner_profile = None
    return adapter


def _message(*, user_id="profile-a-owner-123", chat_type="private", chat_id="chat-a"):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user_id, username="owner"),
        chat=SimpleNamespace(id=chat_id, type=chat_type, is_forum=chat_type == "supergroup"),
        message_thread_id=42 if chat_type == "supergroup" else None,
        is_topic_message=chat_type == "supergroup",
        sender_chat=None,
    )


class _RaisingRunner:
    def _is_user_authorized(self, _source):
        raise RuntimeError("central authorization failed")


class _ReturningRunner:
    def __init__(self, value):
        self.value = value

    def _is_user_authorized(self, _source):
        return self.value


def test_resolver_failure_does_not_inherit_profile_a_authorization(
    monkeypatch, caplog
):
    """A resolver failure for B must not reuse A's global allowlist."""
    global_value = "profile-a-owner-123"
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", global_value)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    original_resolver = authz_mixin._platform_gate_env

    def resolver(name, default=""):
        scope = secret_scope.current_secret_scope() or {}
        if scope.get("PROFILE") == "B":
            raise RuntimeError("resolver failure for profile B")
        return original_resolver(name, default)

    monkeypatch.setattr(authz_mixin, "_platform_gate_env", resolver)
    adapter = _telegram_adapter()

    with caplog.at_level(logging.DEBUG):
        token_a = secret_scope.set_secret_scope(
            {"PROFILE": "A", "TELEGRAM_ALLOWED_USERS": global_value}
        )
        try:
            assert _scoped_gate_env("TELEGRAM_ALLOWED_USERS") == global_value
            assert adapter._is_callback_user_authorized(global_value) is True
        finally:
            secret_scope.reset_secret_scope(token_a)

        token_b = secret_scope.set_secret_scope({"PROFILE": "B"})
        try:
            # RED before the fix: _scoped_gate_env catches the resolver error
            # and returns the process-global allowlist from profile A.
            assert _scoped_gate_env("TELEGRAM_ALLOWED_USERS") == ""
            assert adapter._is_callback_user_authorized(global_value) is False
        finally:
            secret_scope.reset_secret_scope(token_b)

    assert global_value not in caplog.text
    assert "resolver failure for profile B" not in caplog.text


def test_callback_authorizer_exception_cannot_use_permissive_allowlist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "profile-a-owner-123")
    adapter = _telegram_adapter()
    adapter._message_handler = _RaisingRunner()._is_user_authorized

    assert adapter._is_callback_user_authorized("profile-a-owner-123") is False


def test_callback_authorizer_exception_cannot_use_permissive_allow_all(monkeypatch):
    monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    adapter = _telegram_adapter()
    adapter._message_handler = _RaisingRunner()._is_user_authorized

    assert adapter._is_callback_user_authorized("profile-a-owner-123") is False


def test_message_authorizer_exception_cannot_use_permissive_allowlist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "profile-a-owner-123")
    adapter = _message_adapter(_RaisingRunner()._is_user_authorized)

    assert adapter._is_user_authorized_from_message(_message()) is False


def test_message_authorizer_exception_cannot_enter_pairing(monkeypatch):
    monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    adapter = _message_adapter(_RaisingRunner()._is_user_authorized)

    assert adapter._is_user_authorized_from_message(_message()) is False


def test_message_authorizer_false_or_ambiguous_never_uses_env_fallback(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "profile-a-owner-123")
    for result in (False, None):
        adapter = _message_adapter(
            _ReturningRunner(result)._is_user_authorized,
            extra={"unauthorized_dm_behavior": "pair"},
        )
        assert adapter._is_user_authorized_from_message(_message()) is False


def test_registered_authorizer_exception_cannot_use_env_or_pairing(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "profile-a-owner-123")
    adapter = _message_adapter(extra={"unauthorized_dm_behavior": "pair"})
    adapter._authorization_check = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("central callback failed")
    )

    assert adapter._is_user_authorized_from_message(_message()) is False


def test_registered_authorizer_exception_does_not_leak_exception_details(caplog):
    canary = "AUTHZ-CANARY-7f4a"
    absolute_path = "/srv/private/authorization/callback.py"
    callback_url = "https://auth.example.test/check?token=should-not-leak"
    adapter = _message_adapter()

    def raising_authorizer(*_args):
        raise RuntimeError(f"{canary} {absolute_path} {callback_url}")

    adapter._authorization_check = raising_authorizer

    with caplog.at_level(logging.WARNING):
        assert adapter._is_user_authorized_from_message(_message()) is False

    assert canary not in caplog.text
    assert absolute_path not in caplog.text
    assert callback_url not in caplog.text


def test_missing_authorizer_preserves_legacy_pairing(monkeypatch):
    monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
    adapter = _message_adapter(extra={"unauthorized_dm_behavior": "pair"})

    assert adapter._is_user_authorized_from_message(_message()) is True
