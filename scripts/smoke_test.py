"""
End-to-end check of a fresh install, using synthetic data and a throw-away folder + SQLite database:

    uv run python scripts/smoke_test.py                 # ~1-3 minutes, exits 0 on success
    uv run python scripts/smoke_test.py --keep          # keep the temp folder (data, results, .env) to look around
    uv run python scripts/smoke_test.py --port 8123

Steps: make demo data -> start the API -> create a study -> run the pipeline through the API -> check every stage
succeeded and the outputs exist -> run again and check that nothing is recomputed -> stop the API.
Your own .env and database are never touched (all settings are passed as environment variables).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STUDY = "DEMO"           # id of the synthetic study the test creates
EXPECTED_STAGES = 15     # stages of the cohort_with_qc profile; update when a stage is added or removed


def step(msg: str) -> None:
    """Print a numbered progress heading."""
    print(f"\n== {msg}", flush=True)


def fail(msg: str, proc: subprocess.Popen | None = None, log: Path | None = None) -> None:
    """Report the failure with the tail of the server log, stop the server and exit with status 1."""
    print(f"\nFAILED: {msg}", file=sys.stderr)
    if log and log.exists():
        print("--- last lines of the server log ---", file=sys.stderr)
        print("\n".join(log.read_text(errors="replace").splitlines()[-25:]), file=sys.stderr)
    if proc and proc.poll() is None:
        proc.terminate()
    sys.exit(1)


def call(base: str, method: str, path: str, body: dict | list | None = None):
    """One HTTP call to the API. Returns (status code, parsed JSON or raw bytes; the error text for HTTP errors)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw and "json" in r.headers.get("content-type", "") else raw)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def free_port() -> int:
    """A TCP port that is currently unused on localhost."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_study(base: str, timeout: int) -> list[dict]:
    """Start a run through the API and poll until it ends. Returns the per-stage rows; fails the test on a failed stage or a timeout."""
    code, res = call(base, "POST", f"/studies/{STUDY}/run")
    if code != 200:
        fail(f"POST /run -> {code}: {res}")
    run_id = res["run_id"]
    t0, steps = time.time(), []
    while time.time() - t0 < timeout:
        code, steps = call(base, "GET", f"/studies/{STUDY}/runs/{run_id}")
        if code == 200 and steps:
            status = {s["status"] for s in steps}
            if "FAILED" in status:
                bad = [s for s in steps if s["status"] == "FAILED"]
                fail(f"stage failed: {bad[0]['step_name']}: {bad[0]['error_msg']}")
            code, st = call(base, "GET", f"/studies/{STUDY}/status")
            if code == 200 and st["status"] in ("SUCCESS", "COMPLETED", "DONE", "FAILED", "ERROR") and \
                    not status & {"RUNNING", "PENDING"}:
                return steps
        time.sleep(2)
    fail(f"run did not finish within {timeout}s (last steps: {[(s['step_name'], s['status']) for s in steps]})")


def main() -> None:
    """Run the eight steps listed in the module docstring, always cleaning up the server and (unless --keep) the temp folder."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600, help="seconds to wait for each run")
    a = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="ionome-smoke-"))
    port = a.port or free_port()
    base = f"http://127.0.0.1:{port}"
    log = work / "server.log"
    proc = None
    try:
        step(f"1/8 make synthetic data in {work}")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_data.py"), "--out", str(work / "demo"),
                        "--study", STUDY], check=True, stdout=subprocess.DEVNULL)

        env = {**os.environ,
               "PIPELINE_NAME": "smoke", "ENV": "development",
               "LCMS_RAW_DATA_DIR": str(work / "demo" / "raw"),
               "LCMS_RESULTS_DIR": str(work / "results"),
               "LCMS_REFERENCE_LIBRARY_DIR": str(work / "demo" / "library"),
               "YAML_DIR": str(work / "yaml"),
               "LCMS_DB_NAME": "smoke", "LCMS_DB_PATH": str(work / "db" / "smoke.db"),
               "API_HOST": "127.0.0.1", "API_PORT": str(port), "API_RELOAD": "false",
               "LCMS_MAX_WORKERS": "2", "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
        env.pop("API_TOKEN", None)
        (work / ".env.used").write_text("\n".join(f"{k}={v}" for k, v in env.items() if k.startswith(("LCMS_", "API_", "YAML", "PIPE"))))

        step(f"2/8 start the API on {base}")
        proc = subprocess.Popen([sys.executable, "-m", "api"], cwd=work, env=env,
                                stdout=open(log, "w"), stderr=subprocess.STDOUT)
        for _ in range(60):
            if proc.poll() is not None:
                fail("the API exited during start-up", log=log)
            try:
                if call(base, "GET", "/studies")[0] == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            fail("the API did not answer within 60 s", proc, log)
        print("API is up; its database was created at", work / "db" / "smoke.db")

        step("3/8 create the study (what the 'New study' form does)")
        cfg = yaml.safe_load((ROOT / "config" / "experiments" / "template.yaml").read_text())
        cfg["study"] = {"id": STUDY, "polarity": "negative", "instrument": "synthetic"}
        cfg["files"] = {"sample_metadata": f"{STUDY}_sample_metadata.csv"}
        cfg["dataset_profile"] = "cohort_with_qc"
        cfg["library_matching"]["reference_library_path"] = str(work / "demo" / "library" / "demo_library.msp")
        cfg["library_matching"]["reference_library"] = "demo_library"
        cfg["library_matching"]["reference_library_source"] = "synthetic"
        rows = json.loads((work / "demo" / f"{STUDY}_sample_rows.json").read_text())
        code, res = call(base, "POST", "/studies", {"config": cfg, "sample_rows": rows})
        if code != 200:
            fail(f"POST /studies -> {code}: {res}", proc, log)
        print(f"created {STUDY}: {res['ingestion']['n_created']} samples registered")
        code, rd = call(base, "GET", f"/studies/{STUDY}/sample-readiness")
        if code != 200 or rd["missing"]:
            fail(f"raw files missing: {rd}", proc, log)

        step("4/8 run the pipeline (15 stages)")
        t0 = time.time()
        steps = run_study(base, a.timeout)
        names = [s["step_name"] for s in steps]
        print(f"finished in {time.time() - t0:.0f}s; stages: {', '.join(names)}")
        if len(steps) != EXPECTED_STAGES or any(s["status"] != "SUCCESS" for s in steps):
            fail(f"expected {EXPECTED_STAGES} stages all SUCCESS, got {[(s['step_name'], s['status']) for s in steps]}", proc, log)

        step("5/8 check the outputs")
        res_dir = work / "results" / STUDY
        logs = sorted(p.stem for p in (res_dir / "logs").glob("*.json"))
        if len(logs) != EXPECTED_STAGES:
            fail(f"expected {EXPECTED_STAGES} stage logs, found {len(logs)}: {logs}", proc, log)
        for rel in ("resolved_config.json",):
            if not (res_dir / rel).exists():
                fail(f"missing {rel}", proc, log)
        code, prev = call(base, "GET", f"/studies/{STUDY}/results/table/preview")
        if code != 200:
            fail(f"results table not available: {code} {prev}", proc, log)
        code, plots = call(base, "GET", f"/studies/{STUDY}/results/plots")
        if code != 200 or not plots:
            fail(f"no plots listed: {code} {plots}", proc, log)
        qc = json.loads((res_dir / "logs" / "library_matching.json").read_text())["qc_metrics"]
        print(f"table preview OK; {len(plots)} plots; library_matching keys: {sorted(k for k in qc if k != 'results')[:6]}")

        step("6/8 check the results against what the synthetic data contains")
        def metrics(stage):   # qc_metrics of one stage, read from its log file
            return json.loads((res_dir / "logs" / f"{stage}.json").read_text())["qc_metrics"]
        problems = []
        asm = metrics("library_assembly")
        if asm["n_features"] != 16:
            problems.append(f"expected 16 aligned features (12 compounds + 3 unknowns + 1 contaminant), got {asm['n_features']}")
        if asm["n_identified"] != 12:
            problems.append(f"expected all 12 library compounds identified, got {asm['n_identified']}")
        if metrics("blank_qc")["n_flagged"] < 1:
            problems.append("the contaminant that is also in the solvent blank was not flagged")
        vol = metrics("volcano_plot")
        ab, ac = set(vol["A_vs_B"]["significant_feature_ids"]), set(vol["A_vs_C"]["significant_feature_ids"])
        if len(ab) < 3 or len(ac) < 3 or ab & ac:
            problems.append(f"expected the 3 compounds raised in B and the 3 raised in C to be significant and distinct, got A_vs_B={sorted(ab)} A_vs_C={sorted(ac)}")
        if problems:
            fail("; ".join(problems), proc, log)
        print("16 features, 12/12 identified, blank contaminant flagged, B-up and C-up compounds found by the volcano test")

        step("7/8 run again: nothing should be recomputed")
        cps = sorted((res_dir / "checkpoints").glob("*.json"))
        before = {p: p.stat().st_mtime_ns for p in cps}
        if not before:
            fail("no cache checkpoints were written", proc, log)
        run_study(base, a.timeout)
        changed = [p.name for p, t in before.items() if p.stat().st_mtime_ns != t]
        # stages that do not support caching rewrite nothing here; only cached stages have checkpoints
        if changed:
            fail(f"these cached stages were recomputed on the second run: {changed}", proc, log)
        print(f"{len(before)} cached stages untouched on the second run")

        step("8/8 stop the API")
        print("\nSMOKE TEST PASSED")
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(15)
            except subprocess.TimeoutExpired:
                proc.kill()
        if a.keep:
            print(f"\nkept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
