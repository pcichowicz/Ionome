"""
The feature detector must honour `feature_detection.method` and write files the rest of the pipeline can read back.

Guards against the detector being wired to only one algorithm (the chromatogram options in the config then do nothing)
and against the feature schema rejecting the extra fields the chromatogram method writes.
"""
import math

import pytest

import synth_raw
from analysis.utils.json_reader import JSONFeatureReader
from analysis.utils.mzml_parsing import PyteomicsFeatureDetector

PARAMS = {"peak_width": [3, 40], "mass_error_ppm": 5.0, "max_gaps": 2, "min_scans": 2,
          "noise_threshold": 100.0, "polarity": "negative"}


def _raw(tmp_path):
    """60 one-second MS1 scans with one Gaussian peak at m/z 150.0 (apex 10 000 at 30 s) and a flat ion at 300.0."""
    scans = []
    for i in range(60):
        rt = float(i)
        height = 10000.0 * math.exp(-((rt - 30.0) ** 2) / (2 * 4.0 ** 2))
        peaks = [(300.0, 150.0)]
        if height > 1:
            peaks.insert(0, (150.0, height))
        scans.append((1, rt, peaks, None))
    return synth_raw.write_mzml(tmp_path / "s.mzML", scans)


def test_chromatogram_method_is_used_and_readable(tmp_path):
    """The default detector finds the planted peak and reports area and diagnostics."""
    det = PyteomicsFeatureDetector()
    out = tmp_path / "s.featurejson"
    feats = det.detect_features(_raw(tmp_path), out, {**PARAMS, "method": "chromatogram"})
    peak = [f for f in feats if abs(f["mz"] - 150.0) < 0.01]
    assert len(peak) == 1 and abs(peak[0]["rt"] - 30.0) < 3 and peak[0]["charge"] == -1
    assert peak[0]["area"] is not None                         # only the chromatogram method writes this
    assert det.last_diagnostics["method"] == "chromatogram" and "n_traces" in det.last_diagnostics
    back = JSONFeatureReader().read_features(out)               # extra fields must survive the schema check
    assert any(f["area"] is not None for f in back)


def test_legacy_method_still_available(tmp_path):
    """method='legacy' still works and does not write chromatogram-only fields."""
    det = PyteomicsFeatureDetector()
    feats = det.detect_features(_raw(tmp_path), tmp_path / "l.featurejson", {**PARAMS, "method": "legacy"})
    assert any(abs(f["mz"] - 150.0) < 0.01 for f in feats)
    assert all("area" not in f for f in feats)
    assert det.last_diagnostics == {"method": "legacy"}


def test_default_method_is_chromatogram(tmp_path):
    """Without a method the chromatogram detector runs."""
    det = PyteomicsFeatureDetector()
    det.detect_features(_raw(tmp_path), tmp_path / "d.featurejson", dict(PARAMS))
    assert det.last_diagnostics["method"] == "chromatogram"


def test_unknown_method_is_rejected(tmp_path):
    """An unknown method name raises ValueError."""
    with pytest.raises(ValueError, match="method"):
        PyteomicsFeatureDetector().detect_features(_raw(tmp_path), tmp_path / "x.featurejson", {**PARAMS, "method": "nope"})
