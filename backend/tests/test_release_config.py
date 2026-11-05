from pathlib import Path
from app.core.config import Settings, warn_unknown_env_keys


def test_unknown_env_keys_warn_without_values(tmp_path, caplog):
    path = tmp_path / ".env"
    path.write_text("LLM_API_KEY=secret-do-not-log\nLLM_MAX_CONCURENCY=4\n", encoding="utf-8")
    assert warn_unknown_env_keys(path) == ["LLM_MAX_CONCURENCY"]
    assert "LLM_MAX_CONCURENCY" in caplog.text
    assert "secret-do-not-log" not in caplog.text


def test_example_covers_exact_settings_fields():
    from dotenv import dotenv_values
    path = Path(__file__).parents[1] / ".env.example"
    assert {key.lower() for key in dotenv_values(path)} == set(Settings.model_fields)


def test_cors_is_configured():
    from app.main import app
    from app.core.config import settings
    cors = next(item for item in app.user_middleware if item.cls.__name__ == "CORSMiddleware")
    assert cors.kwargs["allow_origins"] == settings.cors_origins
