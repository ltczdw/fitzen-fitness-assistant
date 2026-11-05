from app.core.config import Settings
from app.services.context.budget import ContextBudgetManager


def test_project_runtime_window_and_budget(monkeypatch,tmp_path):
    monkeypatch.delenv('LLM_CONTEXT_WINDOW_TOKENS',raising=False)
    monkeypatch.chdir(tmp_path)
    # Synthetic legacy configuration; never reads the real credentials file.
    legacy=tmp_path/'old.env'
    legacy.write_text('LLM_CONTEXT_WINDOW_TOKENS=32768\nLLM_MAX_OUTPUT_TOKENS=4096\nCONTEXT_SAFETY_MARGIN=0.15\n',encoding='utf-8')
    config=Settings(_env_file=legacy)
    assert config.llm_context_window_tokens==65536
    budget=ContextBudgetManager(window=config.llm_context_window_tokens,
        output=config.llm_max_output_tokens,margin=config.context_safety_margin)
    assert budget.input_budget==51609 and budget.output==4096


def test_explicit_environment_and_constructor_still_override(monkeypatch):
    monkeypatch.setenv('LLM_CONTEXT_WINDOW_TOKENS','98304')
    assert Settings(_env_file=None).llm_context_window_tokens==98304
    assert Settings(_env_file=None,llm_context_window_tokens=49152).llm_context_window_tokens==49152
