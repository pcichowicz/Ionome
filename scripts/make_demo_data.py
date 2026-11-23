"""
Writes a small SYNTHETIC cohort study so the whole pipeline can be tried without real data.

    python scripts/make_demo_data.py --out demo_data            # -> demo_data/raw/DEMO/*.mzXML, library, metadata
    python scripts/make_demo_data.py --out demo_data --study MYDEMO --seed 7

What it makes (all numbers are invented; compound names are "Demo compound NN"):
  * 12 library compounds with an MS2 spectrum each (demo_data/library/demo_library.msp)
  * 1 solvent blank, 3 pooled QC, and 3 groups (A, B, C) x 3 replicates = 13 mzXML files
  * compounds 1-3 are ~4x higher in group B, compounds 4-6 ~4x higher in group C, 7-12 unchanged,
    one contaminant peak appears everywhere including the blank, and a few unknown features have no library entry
The output exercises every cohort stage and gives volcano/PLS-DA something to find. It says nothing about real data.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import struct
from pathlib import Path

import numpy as np

# Size and shape of the synthetic run
N_COMPOUNDS = 12
RUN_SEC = 300          # run length in seconds; one MS1 scan per second
SIGMA_SEC = 6.0        # peak width (FWHM ~ 14 s, inside the default 10-60 s window)
BASE_INTENSITY = 3.0e5
GROUPS = {"A": {}, "B": {1: 4, 2: 4, 3: 4}, "C": {4: 4, 5: 4, 6: 4}}   # compound index -> fold change


# Real molecular formulas, so the library's formula-vs-precursor consistency check passes. Names are generic and the
# spectra are random: these are not real reference spectra.
FORMULAS = ["C6H12O6", "C3H7NO2", "C6H8O7", "C5H5N5", "C4H6O5", "C9H11NO3",
            "C10H13N5O4", "C5H10N2O3", "C6H13NO2", "C9H11NO2", "C4H8N2O3", "C7H6O3"]
_MASS = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956, "P": 30.97376163, "S": 31.972071}
PROTON = 1.00727646688


def neutral_mass(formula: str) -> float:
    """Monoisotopic mass of a formula such as "C6H12O6" (only the elements in _MASS)."""
    import re
    return sum(_MASS[el] * (int(n) if n else 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula))


def compounds(rng: np.random.Generator):
    """The 12 library compounds: [M-H]- m/z from the formula, a retention time every 20 s (with jitter) and random fragments."""
    out = []
    for i in range(1, N_COMPOUNDS + 1):
        formula = FORMULAS[i - 1]
        mz = round(neutral_mass(formula) - PROTON, 4)               # [M-H]-
        rt = round(30 + 20 * i + rng.uniform(-3, 3), 1)
        frags = sorted({round(mz * f + rng.uniform(-1, 1), 4) for f in (0.3, 0.45, 0.55, 0.65, 0.8, 0.9)})
        inten = [round(float(x), 1) for x in rng.uniform(10, 100, len(frags))]
        out.append({"name": f"Demo compound {i:02d}", "formula": formula, "mz": mz, "rt": rt, "frags": list(zip(frags, inten))})
    return out


def write_library(path: Path, comps) -> None:
    """Write the compounds as an MSP reference library (precursor plus fragments, negative mode)."""
    blocks = []
    for c in comps:
        peaks = c["frags"] + [(c["mz"], 100.0)]
        blocks.append(
            f"Name: {c['name']}\nPrecursorMz: {c['mz']}\nPrecursor_type: [M-H]-\nIonMode: negative\n"
            f"RetentionTime: {c['rt'] / 60:.3f}\nFormula: {c['formula']}\nNum peaks: {len(peaks)}\n"
            + "\n".join(f"{mz} {i}" for mz, i in peaks))
    path.write_text("\n\n".join(blocks) + "\n")


def _peak(mz_center, intensity, rng):
    """One measured peak: m/z scattered by ~0.5 ppm, intensity by ~5 %."""
    return mz_center * (1 + rng.normal(0, 0.5e-6)), intensity * rng.lognormal(0, 0.05)


def simulate_sample(comps, group, rng, *, blank=False, qc=False):
    """-> list of scans (ms_level, rt_sec, [(mz, intensity)], precursor_mz|None)"""
    scale = rng.lognormal(0, 0.08)                       # injection-to-injection variation
    unknowns = [(402.1234, 120.0, 0.6), (515.2201, 210.0, 0.4), (333.3333, 80.0, 0.5)]
    contaminant = (279.1591, 150.0, 0.5)                 # in every sample including the blank
    scans = []
    for t in range(RUN_SEC):
        peaks = []
        for k, c in enumerate(comps, start=1):
            if blank:
                continue
            fold = 1.0 if qc else GROUPS[group].get(k, 1)
            if qc:
                fold = 2.0 if k in (1, 2, 3, 4, 5, 6) else 1.0   # pooled QC: mixture of groups
            h = BASE_INTENSITY * fold * scale * np.exp(-0.5 * ((t - c["rt"]) / SIGMA_SEC) ** 2)
            if h > 600:
                peaks.append(_peak(c["mz"], h, rng))
        for mz, rt, rel in unknowns:
            if not blank:
                h = BASE_INTENSITY * rel * scale * np.exp(-0.5 * ((t - rt) / SIGMA_SEC) ** 2)
                if h > 600:
                    peaks.append(_peak(mz, h, rng))
        mz, rt, rel = contaminant
        h = BASE_INTENSITY * rel * (1.0 if blank else 0.3) * np.exp(-0.5 * ((t - rt) / SIGMA_SEC) ** 2)
        if h > 600:
            peaks.append(_peak(mz, h, rng))
        for _ in range(25):                               # chemical noise, below the default 8000 threshold
            peaks.append((float(rng.uniform(100, 700)), float(rng.uniform(300, 2500))))
        scans.append((1, float(t), sorted(peaks), None))
        if not blank:                                     # MS2 on each compound for 3 scans around its apex
            for c in comps:
                if abs(t - c["rt"]) <= 3 and t % 2 == 0:
                    frag = [(m, i * 50 * rng.lognormal(0, 0.05)) for m, i in c["frags"] + [(c["mz"], 100.0)]]
                    scans.append((2, t + 0.5, sorted(frag), c["mz"]))
    return scans


def write_mzxml(path: Path, scans) -> None:
    """Write scans as a minimal mzXML 3.2 file (64-bit big-endian peaks, uncompressed) that the pipeline's reader accepts."""
    out = []
    for i, (level, rt, peaks, prec) in enumerate(scans, start=1):
        flat = [v for p in peaks for v in p]
        raw = base64.b64encode(struct.pack(f">{len(flat)}d", *flat)).decode()
        pm = f'<precursorMz precursorIntensity="1000">{prec}</precursorMz>' if prec is not None else ""
        out.append(f'<scan num="{i}" msLevel="{level}" peaksCount="{len(peaks)}" retentionTime="PT{rt}S">{pm}'
                   f'<peaks precision="64" byteOrder="network" contentType="m/z-int" compressionType="none" '
                   f'compressedLen="0">{raw}</peaks></scan>')
    path.write_text('<?xml version="1.0" encoding="ISO-8859-1"?>\n'
                    '<mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2">'
                    f'<msRun scanCount="{len(scans)}">\n' + "\n".join(out) + "\n</msRun></mzXML>\n")


def main() -> None:
    """Write the raw files, library, sample sheet and a ready-to-paste .env block."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="demo_data", help="output folder (default: demo_data)")
    ap.add_argument("--study", default="DEMO", help="study id (default: DEMO)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--write-env", action="store_true",
                    help="also write the settings below to <repo>/.env (only if that file does not exist yet)")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    raw = out / "raw" / a.study
    lib = out / "library"
    raw.mkdir(parents=True, exist_ok=True)
    lib.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    comps = compounds(rng)
    write_library(lib / "demo_library.msp", comps)

    rows = [("blank_1", "solvent_blank", None)]
    rows += [(f"QC_{i}", "QC", "qc") for i in (1, 2, 3)]
    rows += [(f"{g}_{i}", g, g) for g in GROUPS for i in (1, 2, 3)]
    for sample_id, role, grp in rows:
        scans = simulate_sample(comps, grp, rng, blank=role == "solvent_blank", qc=role == "QC")
        write_mzxml(raw / f"{sample_id}.mzXML", scans)
        print("wrote", raw / f"{sample_id}.mzXML")

    with open(raw / f"{a.study}_sample_metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "sample_role"])
        w.writerows([(s, r) for s, r, _ in rows])
    (out / f"{a.study}_sample_rows.json").write_text(
        json.dumps([{"sample_id": s, "sample_role": r} for s, r, _ in rows], indent=2))
    print(f"\nlibrary : {lib / 'demo_library.msp'}\nmetadata: {raw / (a.study + '_sample_metadata.csv')}")

    work = out / "work"
    env_text = "\n".join([
        "PIPELINE_NAME=ionome-lcms",
        f"LCMS_RAW_DATA_DIR={out / 'raw'}",          # the folder that CONTAINS the study folder
        f"LCMS_RESULTS_DIR={work / 'results'}",
        f"LCMS_REFERENCE_LIBRARY_DIR={lib}",
        f"YAML_DIR={work / 'yaml'}",
        "LCMS_DB_NAME=ionome",
        f"LCMS_DB_PATH={work / 'ionome.db'}",
        "API_HOST=127.0.0.1",
        "API_PORT=8000",
        "API_RELOAD=false",
    ]) + "\n"
    env_path = Path(__file__).resolve().parents[1] / ".env"
    print("\nThe app reads its settings from a .env file in the repo root. For this demo it should contain:\n")
    print(env_text)
    if a.write_env and not env_path.exists():
        env_path.write_text(env_text)
        print(f"wrote {env_path}")
    elif a.write_env:
        print(f"{env_path} already exists, left unchanged: compare it with the block above.")
    else:
        print(f"Paste that into {env_path}, or re-run with --write-env to have it written for you.")


if __name__ == "__main__":
    main()
