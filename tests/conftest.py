"""Global test isolation from developer credentials and study configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_dotenv_autoload(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
