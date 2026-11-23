"""
API tests run against a throw-away data directory and database. The settings object is created when
core.settings is first imported, so the environment has to be set here, before any test module imports the app.
"""
import os
import sys
import tempfile
from pathlib import Path

_root = Path(tempfile.mkdtemp(prefix="ionome-test-"))
for sub in ("raw", "results", "yaml", "refs", "db"):
    (_root / sub).mkdir()
os.environ.update({
    "PIPELINE_NAME": "test",
    "LCMS_RAW_DATA_DIR": str(_root / "raw"),
    "LCMS_RESULTS_DIR": str(_root / "results"),
    "LCMS_REFERENCE_LIBRARY_DIR": str(_root / "refs"),
    "YAML_DIR": str(_root / "yaml"),
    "OUTPUT_DIRS": '["features","logs","plots","library","qc_reports","cache"]',
    "LCMS_DB_NAME": "test",
    "LCMS_DB_PATH": str(_root / "db" / "test.db"),
    "API_RELOAD": "false",
    "API_HOST": "127.0.0.1",
    "API_ALLOWED_HOSTS": "localhost,127.0.0.1,testserver",
    "LCMS_MAX_WORKERS": "2",
    "ENV": "development",
})
os.environ.pop("API_TOKEN", None)
sys.path.insert(0, str(Path(__file__).parent))      # fake_runner.py, importable by spawned workers too
TEST_ROOT = _root
