from pathlib import Path
import pytest
from app.db import repositories, session as db


def test_online_repository_has_no_json_writer():
    assert not hasattr(repositories, "_write_fallback")
    assert not hasattr(repositories, "FALLBACK_PATH")


def test_real_data_directory_rejects_writes():
    target = Path(__file__).resolve().parents[1] / "data" / "isolation-probe.json"
    with pytest.raises(AssertionError, match="must not write"):
        target.write_text("[]", encoding="utf-8")
    assert not target.exists()


def test_live_database_is_not_available_to_unit_tests():
    assert db.DB_READY is False
    assert db.SessionLocal is None
