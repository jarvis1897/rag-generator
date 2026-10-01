import logging

import pytest

from app.config import Secrets, Settings, get_settings
from app.llm import LLMError, create_llm


def test_api_key_from_env_file_reaches_client(tmp_path, monkeypatch) -> None:
    # pydantic-settings reads .env without exporting it, so the key must be passed explicitly.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY = sk-test-123\n")
    secrets = Secrets(_env_file=env)
    assert "sk-test-123" not in repr(secrets)  # SecretStr keeps it out of logs
    llm = create_llm(Settings(), secrets)
    assert llm._client.api_key == "sk-test-123"


def test_missing_credentials_become_llm_error(monkeypatch, tmp_path) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))  # no `ant auth login` profile
    # Depending on SDK credential resolution this fails at construction or at the first call.
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        create_llm(Settings(), Secrets(_env_file=None)).complete(
            "system", [{"role": "user", "content": "hi"}], max_tokens=10
        )


def test_missing_credentials_without_profile_dir(monkeypatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    try:
        llm = create_llm(Settings(), Secrets(_env_file=None))
    except LLMError:
        return
    if llm._client.api_key or llm._client.auth_token:
        pytest.skip("credentials available on this machine")
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        llm.complete("system", [{"role": "user", "content": "hi"}], max_tokens=10)


# --- single source of configuration ---


def test_settings_ignore_environment_and_env_file(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("CHUNK_SIZE=99\nANTHROPIC_API_KEY=sk-x\n")
    monkeypatch.setenv("TOP_K", "42")
    get_settings.cache_clear()
    try:
        with caplog.at_level(logging.WARNING, logger="app.config"):
            settings = get_settings()
    finally:
        get_settings.cache_clear()
    assert settings.chunk_size == Settings.model_fields["chunk_size"].default
    assert settings.top_k == Settings.model_fields["top_k"].default
    warnings = " ".join(r.getMessage() for r in caplog.records)
    assert "CHUNK_SIZE" in warnings and "TOP_K" in warnings
    assert "ANTHROPIC_API_KEY" not in warnings  # secrets belong in .env


def test_settings_reject_unknown_fields() -> None:
    with pytest.raises(ValueError):
        Settings(chunk_sise=100)
