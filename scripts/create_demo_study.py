"""
Registers the synthetic demo study (made by scripts/make_demo_data.py) with a RUNNING API, the same call the
'New study' form makes.

    uv run python scripts/create_demo_study.py --data demo_data                   # study DEMO on http://127.0.0.1:8000
    uv run python scripts/create_demo_study.py --data demo_data --study MYDEMO --url http://127.0.0.1:8011
    uv run python scripts/create_demo_study.py --data demo_data --token SECRET    # if the API has API_TOKEN set

The API must have been started with LCMS_RAW_DATA_DIR pointing at <data>/raw.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Build the study config from config/experiments/template.yaml plus the synthetic library, then POST it to /studies."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="demo_data", help="folder given to make_demo_data.py --out")
    ap.add_argument("--study", default="DEMO", help="study id; must match the --study given to make_demo_data.py")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default=None, help="API_TOKEN, if one is set")
    ap.add_argument("--profile", default="cohort_with_qc", choices=["cohort_with_qc", "standards_only"])
    a = ap.parse_args()

    data = Path(a.data).resolve()
    cfg = yaml.safe_load((ROOT / "config" / "experiments" / "template.yaml").read_text())
    cfg["study"] = {"id": a.study, "polarity": "negative", "instrument": "synthetic"}
    cfg["files"] = {"sample_metadata": f"{a.study}_sample_metadata.csv"}
    cfg["dataset_profile"] = a.profile
    cfg["library_matching"]["reference_library_path"] = str(data / "library" / "demo_library.msp")
    cfg["library_matching"]["reference_library"] = "demo_library"
    cfg["library_matching"]["reference_library_source"] = "synthetic"
    rows = json.loads((data / f"{a.study}_sample_rows.json").read_text())

    headers = {"Content-Type": "application/json"}
    if a.token:
        headers["Authorization"] = "Basic " + base64.b64encode(f":{a.token}".encode()).decode()
    req = urllib.request.Request(a.url + "/studies", data=json.dumps({"config": cfg, "sample_rows": rows}).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            res = json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"the API refused it ({e.code}): {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"could not reach {a.url}: {e.reason}. Is the API running?")
    print(f"created study {res['study_id']}: {res['ingestion']['n_created']} samples registered")
    print(f"next: open {a.url}/studies/{a.study} and press Run, or: uv run python -m analysis.pipeline_orchestrator {a.study}")


if __name__ == "__main__":
    main()
