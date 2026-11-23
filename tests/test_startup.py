"""Starting without (or with a broken) .env gives a short, actionable message instead of a pydantic traceback."""
import runpy

import pytest

import core.settings as settings_mod

REQUIRED = ["PIPELINE_NAME", "LCMS_RAW_DATA_DIR", "LCMS_RESULTS_DIR", "LCMS_REFERENCE_LIBRARY_DIR", "YAML_DIR",
            "LCMS_DB_NAME", "LCMS_DB_PATH", "API_RELOAD"]


@pytest.fixture
def no_settings(monkeypatch):
    """No environment variables and no .env file, whatever the developer's machine has."""
    for name in REQUIRED:
        monkeypatch.delenv(name, raising=False)

    class NoFile(settings_mod.Settings):
        model_config = {**settings_mod.Settings.model_config, "env_file": None}

    monkeypatch.setattr(settings_mod, "Settings", NoFile)
    monkeypatch.setattr(settings_mod, "load_dotenv", lambda *a, **k: None)
    settings_mod.reset_settings()
    yield
    settings_mod.reset_settings()


def test_missing_settings_are_listed_with_the_fix(no_settings):
    """A missing .env gives a readable list of missing settings and the fix, not a pydantic traceback."""
    with pytest.raises(settings_mod.SettingsError) as e:
        settings_mod.get_settings()
    text = str(e.value)
    for name in REQUIRED:
        assert name in text
    assert "cp .env.example .env" in text or "Fix or add the values" in text
    assert "validation error" not in text.lower() and "pydantic" not in text.lower()


def test_python_m_api_prints_the_message_and_exits_2(no_settings, capsys):
    """`python -m api` without settings prints that message and exits with status 2."""
    with pytest.raises(SystemExit) as e:
        runpy.run_module("api", run_name="__main__")
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "Ionome cannot start" in err and "PIPELINE_NAME" in err and "Traceback" not in err


def test_demo_data_prints_a_usable_env_block(tmp_path, capsys, monkeypatch):
    """make_demo_data prints a .env block that is itself a valid configuration."""
    import sys
    sys.path.insert(0, str(settings_mod.PROJECT_ROOT / "scripts"))
    import make_demo_data
    monkeypatch.setattr(sys, "argv", ["make_demo_data.py", "--out", str(tmp_path / "d")])
    make_demo_data.main()
    out = capsys.readouterr().out
    block = out.split("should contain:")[1].split("Paste that")[0]
    env = dict(line.split("=", 1) for line in block.strip().splitlines())
    assert set(REQUIRED) <= set(env)
    from core.settings import Settings
    s = Settings(_env_file=None, **{k: v for k, v in env.items()})          # the block really is a valid configuration
    assert s.lcms_raw_data_dir == (tmp_path / "d" / "raw").resolve()
