import sys
from pathlib import Path

import pytest

# Find the repo root (the folder holding tools/ and student/) however deeply the tests are nested.
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tools").is_dir() and (p / "student").is_dir())
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Every test gets a fresh scheduler database."""
    monkeypatch.setenv("CAMPUSAI_DB", str(tmp_path / "test.db"))
