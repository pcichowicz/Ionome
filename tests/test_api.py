"""API tests: input hardening, authentication, browser security headers and the run lifecycle (using a fake pipeline runner)."""
import json
import os
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from conftest import TEST_ROOT
from api.main import app
from api.services import runs as runs_service
from core.db import SyncSessionLocal
from core.models import Experiment, Pipeline, Sample
from core.settings import settings
from core.status import STATUS

RUNNER = "fake_runner:run"
HEADERS = {"host": "localhost"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://localhost") as c:      # `with` runs the lifespan (creates tables)
        yield c
    runs_service.shutdown_pool()


def _config(study_id: str, **over) -> dict:
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/experiments/template.yaml").read_text())
    cfg["study"] = {"id": study_id, "polarity": "negative", "instrument": "Orbitrap"}
    cfg["files"] = {"sample_metadata": f"{study_id}_sample_metadata.csv"}
    cfg.update(over)
    return cfg


def _make_study(client, study_id: str, sample_ids=("s1", "s2")) -> dict:
    raw = TEST_ROOT / "raw" / study_id
    raw.mkdir(parents=True, exist_ok=True)
    for sid in sample_ids:
        (raw / f"{sid}.mzML").write_text("x")
    return client.post("/studies", json={"config": _config(study_id),
                                         "sample_rows": [{"sample_id": s, "sample_role": "sample"} for s in sample_ids]})


# A run happens in a freshly spawned worker process. On a cold machine (first run after install, antivirus or
# Gatekeeper scanning the new libraries, a laptop on battery) that alone can take tens of seconds, so the limit is
# generous and can be raised: IONOME_TEST_TIMEOUT=300 uv run pytest
WAIT_TIMEOUT = float(os.environ.get("IONOME_TEST_TIMEOUT", "120"))


def _wait_for(client, study_id: str, status: str, timeout: float | None = None):
    timeout = WAIT_TIMEOUT if timeout is None else timeout
    end = time.time() + timeout
    last = None
    while time.time() < end:
        last = client.get(f"/studies/{study_id}/status").json()["status"]
        if last == status:
            return
        time.sleep(0.1)
    samples = [x["status"] for x in client.get(f"/studies/{study_id}/samples").json()]
    raise AssertionError(f"{study_id} never reached {status} within {timeout:.0f}s (last status: {last}; "
                         f"sample statuses: {samples}). "
                         f"If the machine is just slow, raise IONOME_TEST_TIMEOUT.")


def _outside_files() -> set[Path]:
    return {p for p in TEST_ROOT.rglob("*") if p.is_file() and TEST_ROOT / "raw" not in p.parents
            and TEST_ROOT / "yaml" not in p.parents and TEST_ROOT / "db" not in p.parents
            and TEST_ROOT / "results" not in p.parents}


# ------------------------------------------------------------------ path handling

@pytest.mark.parametrize("bad", ["..", "a..b/c", "%2e%2e", "-x/", "x" * 80, ".hidden"])
def test_raw_files_rejects_bad_study_ids(client, bad):
    """A study id with path characters is refused on the raw-file scan."""
    assert client.get(f"/studies/{bad}/raw-files").status_code in (404, 422)


def test_create_rejects_traversal_in_study_id(client):
    """Creating a study with a path-like id fails with 400 and writes nothing outside the data folders."""
    before = _outside_files()
    for bad in ["../evil", "/tmp/evil", "a/b", "..", ""]:
        r = client.post("/studies", json={"config": _config(bad), "sample_rows": [{"sample_id": "s1"}]})
        assert r.status_code == 400, (bad, r.text)
    assert _outside_files() == before and not (TEST_ROOT / "evil").exists()


def test_create_rejects_metadata_file_that_leaves_the_study_folder(client):
    """The sample-sheet name must be the study's own CSV; paths, other extensions and other studies' names are refused."""
    (TEST_ROOT / "raw" / "trav1").mkdir(exist_ok=True)
    (TEST_ROOT / "raw" / "trav1" / "s1.mzML").write_text("x")
    for bad in ["../../pwned.csv", "/tmp/pwned.csv", "sub/x.csv", "x.txt", "trav2_sample_metadata.csv"]:
        cfg = _config("trav1"); cfg["files"] = {"sample_metadata": bad}
        r = client.post("/studies", json={"config": cfg, "sample_rows": [{"sample_id": "s1"}]})
        assert r.status_code == 400, (bad, r.text)
    assert not list(TEST_ROOT.rglob("pwned.csv")) and not Path("/tmp/pwned.csv").exists()


def test_create_rejects_unsafe_and_duplicate_sample_ids(client):
    """Sample ids that are unsafe as file names, or repeated, are refused with a clear message."""
    (TEST_ROOT / "raw" / "trav3").mkdir(exist_ok=True)
    for bad in ["../x", "/abs/x", "a\\b", ".hidden", "a..b"]:
        r = client.post("/studies", json={"config": _config("trav3"), "sample_rows": [{"sample_id": bad}]})
        assert r.status_code == 400 and "not allowed in file names" in json.dumps(r.json()), (bad, r.text)
    r = client.post("/studies", json={"config": _config("trav3"), "sample_rows": [{"sample_id": "s1"}, {"sample_id": "s1"}]})
    assert r.status_code == 400 and "duplicate" in json.dumps(r.json())


def test_find_raw_file_refuses_paths():
    """find_raw_file never returns a file outside the given folder, even for ids like '../x' or absolute paths."""
    from core.ingestion import find_raw_file
    (TEST_ROOT / "raw" / "ff").mkdir(exist_ok=True)
    (TEST_ROOT / "raw" / "outside.mzML").write_text("x")
    assert find_raw_file(TEST_ROOT / "raw" / "ff", "../outside") is None
    assert find_raw_file(TEST_ROOT / "raw" / "ff", str(TEST_ROOT / "raw" / "outside")) is None


def test_valid_study_still_created(client):
    """The checks above do not block a normal study: it is created, its YAML written and its samples ingested."""
    r = _make_study(client, "okstudy")
    assert r.status_code == 200, r.text
    assert (TEST_ROOT / "yaml" / "okstudy.yaml").exists()
    assert r.json()["ingestion"]["n_created"] == 2


# ------------------------------------------------------------------ access control

def test_token_required_when_configured(client, monkeypatch):
    """With API_TOKEN set, pages and API calls need the token as the Basic-auth password."""
    monkeypatch.setattr(settings, "api_token", "s3cret")
    assert client.get("/studies").status_code == 401
    assert client.get("/").status_code == 401
    assert client.get("/studies", auth=("anyone", "wrong")).status_code == 401
    assert client.get("/studies", auth=("anyone", "s3cret")).status_code == 200
    assert client.post("/studies/okstudy/run", auth=None).status_code == 401


def test_no_token_means_open_on_loopback(client):
    """Without a token the API is open (allowed because it listens on loopback only)."""
    assert client.get("/studies").status_code == 200


def test_non_loopback_host_needs_a_token():
    """Settings refuse a non-loopback API_HOST unless API_TOKEN is set."""
    from core.settings import Settings
    base = dict(PIPELINE_NAME="x", LCMS_RAW_DATA_DIR="/a", LCMS_RESULTS_DIR="/b", LCMS_REFERENCE_LIBRARY_DIR="/c",
                YAML_DIR="/d", OUTPUT_DIRS=["f"], LCMS_DB_NAME="n", LCMS_DB_PATH="/e/x.db", API_RELOAD=False)
    with pytest.raises(ValueError, match="API_TOKEN"):
        Settings(API_HOST="0.0.0.0", **base)
    Settings(API_HOST="0.0.0.0", API_TOKEN="t", **base)
    Settings(API_HOST="127.0.0.1", **base)


def test_cross_site_posts_are_refused(client):
    """State-changing requests from another origin get 403; same-origin and non-browser clients pass."""
    for extra in ({"origin": "http://evil.example"}, {"sec-fetch-site": "cross-site"}):
        assert client.post("/studies/okstudy/run", headers=extra).status_code == 403
    assert client.post("/studies/nope/run", headers={"origin": "http://localhost"}).status_code == 404   # passed the check
    assert client.post("/studies/nope/run").status_code == 404                                           # curl-style, no Origin


def test_unknown_host_header_refused(client):
    """A request with an unexpected Host header is refused (DNS-rebinding protection)."""
    assert client.get("/studies", headers={"host": "evil.example"}).status_code == 400


def test_browser_hardening_headers(client):
    """Responses carry the CSP and nosniff headers (the interactive docs are exempt from the CSP)."""
    r = client.get("/")
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" not in client.get("/docs").headers      # Swagger UI needs inline scripts


def test_docs_hidden_outside_development(monkeypatch):
    """/docs and /openapi.json are served only when ENV=development."""
    import importlib, api.main
    monkeypatch.setattr(settings, "env", "production")
    app2 = importlib.reload(api.main).app
    try:
        with TestClient(app2, base_url="http://localhost") as c:
            assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404
    finally:
        monkeypatch.setattr(settings, "env", "development")
        importlib.reload(api.main)


# ------------------------------------------------------------------ HTML injection

def test_dashboard_data_cannot_close_the_script_block(client):
    """Study data embedded in the dashboard page cannot break out of its JSON script block."""
    yaml_path = TEST_ROOT / "yaml" / "inject.yaml"
    yaml_path.write_text(yaml.safe_dump({"dataset_profile": "</script><img id=pwned src=x onerror=alert(1)>"}))
    with SyncSessionLocal() as s:
        s.add(Experiment(study_id="inject", config_path=str(yaml_path), status=STATUS.PENDING)); s.commit()
    html = client.get("/").text
    assert "<img id=pwned" not in html and "</script><img" not in html
    block = html.split('id="initial-studies"')[1].split("</script>")[0]
    assert "\\u003c/script\\u003e" in block            # escaped by tojson, still valid JSON for the page


# ------------------------------------------------------------------ runs

def test_second_start_while_running_gets_409_then_run_completes(client):
    """A run request during an active run gets 409; a normal run through the worker pool completes."""
    assert _make_study(client, "slowone").status_code == 200
    assert runs_service.claim_run("slowone") is True            # what POST /run does first
    r = client.post("/studies/slowone/run")
    assert r.status_code == 409 and "already running" in r.json()["detail"]
    runs_service.set_experiment_status("slowone", STATUS.PENDING)
    # real start through the endpoint, using the fake runner
    from api.endpoints import runs as runs_endpoint
    orig = runs_endpoint.submit_run
    runs_endpoint.submit_run = lambda sid, pk, rid, runner=RUNNER: orig(sid, pk, rid, RUNNER)
    try:
        first = client.post("/studies/slowone/run"); second = client.post("/studies/slowone/run")
        assert first.status_code == 200 and second.status_code == 409
        _wait_for(client, "slowone", "SUCCESS")
    finally:
        runs_endpoint.submit_run = orig
    steps = client.get(f"/studies/slowone/runs/{first.json()['run_id']}").json()
    assert [(s["step_name"], s["status"]) for s in steps] == [("ingestion", "SUCCESS"), ("feature_detection", "SUCCESS")]
    assert {s["status"] for s in client.get("/studies/slowone/samples").json()} == {"SUCCESS"}


def _start(client, study_id):
    from api.endpoints import runs as runs_endpoint
    orig = runs_endpoint.submit_run
    runs_endpoint.submit_run = lambda sid, pk, rid, runner=RUNNER: orig(sid, pk, rid, RUNNER)
    try:
        return client.post(f"/studies/{study_id}/run")
    finally:
        runs_endpoint.submit_run = orig


def test_pipeline_error_is_recorded_with_its_message(client):
    """A stage that raises marks the run and study FAILED and stores the error message; the study can be run again."""
    assert _make_study(client, "boomone").status_code == 200
    run_id = _start(client, "boomone").json()["run_id"]
    _wait_for(client, "boomone", "FAILED")
    steps = {s["step_name"]: s for s in client.get(f"/studies/boomone/runs/{run_id}").json()}
    assert steps["feature_detection"]["status"] == "FAILED" and "detector exploded" in steps["feature_detection"]["error_msg"]
    assert "RuntimeError: detector exploded" in steps["pipeline"]["error_msg"]
    assert client.post("/studies/boomone/run").status_code in (200, 503)      # a failed study can be started again


def test_killed_worker_marks_run_failed_and_pool_recovers(client):
    """If the worker process dies the run is marked FAILED and later runs still work (the pool is rebuilt)."""
    assert _make_study(client, "killone").status_code == 200
    assert _make_study(client, "afterkill").status_code == 200
    run_id = _start(client, "killone").json()["run_id"]
    _wait_for(client, "killone", "FAILED")
    steps = {s["step_name"]: s for s in client.get(f"/studies/killone/runs/{run_id}").json()}
    assert "worker process died" in steps["pipeline"]["error_msg"]
    assert {s["status"] for s in client.get("/studies/killone/samples").json()} == {"FAILED"}
    assert _start(client, "afterkill").status_code == 200                     # a broken pool must not stick
    _wait_for(client, "afterkill", "SUCCESS")


def test_runs_left_running_by_a_dead_server_are_failed_at_startup(client):
    """At startup, runs still marked RUNNING from a previous server process are set to FAILED."""
    assert _make_study(client, "stale").status_code == 200
    with SyncSessionLocal() as s:
        exp = s.query(Experiment).filter_by(study_id="stale").one()
        exp.status = STATUS.RUNNING
        s.add(Pipeline(run_id="r-old", experiment_id=exp.id, step_name="alignment", status=STATUS.RUNNING))
        s.query(Sample).filter_by(experiment_id=exp.id).update({Sample.status: STATUS.RUNNING}); s.commit()
    assert runs_service.recover_interrupted_runs() >= 1
    assert client.get("/studies/stale/status").json()["status"] == "FAILED"
    step = client.get("/studies/stale/runs/r-old").json()[0]
    assert step["status"] == "FAILED" and "server stopped" in step["error_msg"]
    assert client.post("/studies/stale/run").status_code != 409
