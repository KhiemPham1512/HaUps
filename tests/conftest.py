import subprocess

import pytest


@pytest.fixture(autouse=True)
def never_launch_real_engine_in_unit_tests(monkeypatch, tmp_path):
    import app.lock

    monkeypatch.setattr(app.lock, "ROOT", tmp_path)

    def blocked(*args, **kwargs):
        raise AssertionError(
            "Mock subprocess explicitly; GPU tests run separately with opt-in."
        )

    monkeypatch.setattr(subprocess, "run", blocked)
