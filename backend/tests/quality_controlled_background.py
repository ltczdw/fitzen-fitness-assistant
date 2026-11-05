"""Opt-in test plugin: controlled enabled defaults, explicit off controls still win."""
import os
import pytest


@pytest.fixture(autouse=True)
def controlled_background(monkeypatch, record_property):
    assert os.getenv("FITZEN_CONTROLLED_BACKGROUND") == "1", "controlled test opt-in required"
    from app.core.config import settings
    monkeypatch.setattr(settings, "memory_background_enabled", True)
    monkeypatch.setattr(settings, "memory_summary_enabled", True)
    yield
    record_property("background_at_test_end", str(settings.memory_background_enabled).lower())
    record_property("summary_at_test_end", str(settings.memory_summary_enabled).lower())
