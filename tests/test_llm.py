import pytest

from app.config import Settings
from app.llm import LLMError, create_llm


def test_api_key_from_env_file_reaches_client(tmp_path, monkeypatch) -> None:
    # pydantic-settings reads .env without exporting it, so the key must be passed explicitly.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY = sk-test-123\n")
    settings = Settings(_env_file=env)
    assert "sk-test-123" not in repr(settings)  # SecretStr keeps it out of logs
    llm = create_llm(settings)
    assert llm._client.api_key == "sk-test-123"


def test_missing_credentials_become_llm_error(monkeypatch, tmp_path) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))  # no `ant auth login` profile
    # Depending on SDK credential resolution this fails at construction or at the first call.
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        create_llm(Settings(_env_file=None)).complete("system", [{"role": "user", "content": "hi"}], max_tokens=10)


def test_missing_credentials_without_profile_dir(monkeypatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    try:
        llm = create_llm(Settings(_env_file=None))
    except LLMError:
        return
    if llm._client.api_key or llm._client.auth_token:
        pytest.skip("credentials available on this machine")
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        llm.complete("system", [{"role": "user", "content": "hi"}], max_tokens=10)
