"""
Readers for raw mzML / mzXML files (via pyteomics) and the feature detector entry point.

Both formats are supported everywhere and dispatched by file extension (case-insensitive). Retention times are
always returned in SECONDS (mzML/mzXML may store minutes or seconds; the unit is read from the file).
Main pieces: precursor / MS2 / purity readers, `load_ms1_scans` (MS1 scans as numpy arrays), `build_ms2_index` and
`lookup_best_ms2` (fast MS2 lookup), and `PyteomicsFeatureDetector`, which runs the algorithm chosen by
`feature_detection.method`.
"""
from pyteomics import mzml
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Optional

from analysis.utils.feature_utils import group_into_features
from analysis.utils.peak_detection import detect_features_chromatogram

class PyteomicsPrecursorReader:
    """
    Dual-format (mzML/mzXML) precursor m/z reader. Dispatches by file
    extension, since the two formats use different key structures for
    MS2 precursor information.
    """

    def get_precursor_mzs(self, raw_path: Path) -> list[float]:
        """Precursor m/z of every MS2 scan in the file (one entry per scan, duplicates kept); ValueError for other extensions."""
        suffix = raw_path.suffix.lower()
        if suffix == ".mzml":
            return self._get_precursor_mzs_mzml(raw_path)
        elif suffix == ".mzxml":
            return self._get_precursor_mzs_mzxml(raw_path)
        else:
            raise ValueError(f"Unsupported raw file extension '{suffix}' for {raw_path}")

    def _get_precursor_mzs_mzml(self, mzml_path: Path) -> list[float]:
        """mzML implementation of get_precursor_mzs."""
        from pyteomics import mzml

        precursor_mzs: list[float] = []
        with mzml.read(str(mzml_path)) as reader:
            for spectrum in reader:
                if spectrum.get("ms level") != 2:
                    continue
                for precursor in spectrum.get("precursorList", {}).get("precursor", []):
                    for ion in precursor.get("selectedIonList", {}).get("selectedIon", []):
                        mz = ion.get("selected ion m/z")
                        if mz is not None:
                            precursor_mzs.append(float(mz))
        return precursor_mzs

    def _get_precursor_mzs_mzxml(self, mzxml_path: Path) -> list[float]:
        """mzXML implementation of get_precursor_mzs."""
        from pyteomics import mzxml

        precursor_mzs: list[float] = []
        with mzxml.read(str(mzxml_path)) as reader:
            for spectrum in reader:
                if spectrum.get("msLevel") != 2:
                    continue
                for precursor_entry in spectrum.get("precursorMz", []):
                    mz = precursor_entry.get("precursorMz")
                    if mz is not None:
                        precursor_mzs.append(float(mz))
        return precursor_mzs

class PyteomicsFeatureDetector:
    """
    Detect features in one raw file and write them as JSON (FEATURE_FILE_SUFFIX).

    `params["method"]` selects the algorithm: "chromatogram" (default; analysis.utils.peak_detection) or "legacy"
    (analysis.utils.feature_utils.group_into_features, kept for comparison runs). Diagnostics of the last call are
    kept in `last_diagnostics`, which the feature_detection stage copies into its metrics.
    """

    def __init__(self):
        """Start with empty diagnostics."""
        self.last_diagnostics: dict = {}

    def detect_features(self, mzml_path: Path, output_path: Path, params: dict) -> list[dict]:
        """
        Load the MS1 scans of `mzml_path`, detect features and write them to `output_path`.

        Args:
            mzml_path: .mzML or .mzXML file.
            output_path: where the JSON list of features is written (parent folders are created).
            params: the feature_detection options plus "polarity" (see core.config_schema.FeatureDetection).

        Returns:
            The detected features (dicts with mz, rt in seconds, intensity, charge; the chromatogram method adds
            area, rt_start, rt_end, n_scans and snr).

        Raises:
            ValueError: `params["method"]` is neither "chromatogram" nor "legacy".
        """
        import json

        scans = load_ms1_scans(mzml_path)
        method = str(params.get("method", "chromatogram"))
        if method == "chromatogram":
            features, diag = detect_features_chromatogram(scans, params)
        elif method == "legacy":
            features, diag = self._detect_legacy(scans, params), {"method": "legacy"}
        else:
            raise ValueError(f"feature_detection.method must be 'chromatogram' or 'legacy', got {method!r}")
        diag["method"] = method
        self.last_diagnostics = diag
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(features, f, indent=2)
        return features

    @staticmethod
    def _detect_legacy(scans: list[dict], params: dict) -> list[dict]:
        """The previous detector: group peaks with similar m/z across consecutive scans; only peak_width[0] is used."""
        peaks_width_sec = params.get("peak_width", [10, 60])
        rts = [s["rt"] for s in scans]
        scan_interval_sec = float(np.median(np.diff(sorted(rts))))
        min_scans = max(2, int(round(peaks_width_sec[0] / scan_interval_sec)))
        return group_into_features(
            scans,
            mz_ppm_tolerance=float(params.get("mass_error_ppm", 5.0)),
            min_scans=min_scans,
            noise_threshold=float(params.get("noise_threshold", 8000.0)),
            max_gaps=int(params.get("max_gaps", 2)),
        )

def load_ms1_scans(raw_path: Path) -> list[dict]:
    """
    Pull all MS1 scans out of an mzML or mzXML file as plain dicts of
    numpy arrays, via pyteomics. Dispatches by file extension; both
    branches use the same unit_info-based RT handling, since neither
    format's RT unit should be assumed.
    """
    suffix = raw_path.suffix.lower()
    if suffix == ".mzml":
        from pyteomics import mzml as reader_module
        ms_level_key, scan_time_getter = "ms level", _mzml_scan_time
    elif suffix == ".mzxml":
        from pyteomics import mzxml as reader_module
        ms_level_key, scan_time_getter = "msLevel", _mzxml_scan_time
    else:
        raise ValueError(f"Unsupported raw file extension '{suffix}' for {raw_path}")

    scans = []
    with reader_module.read(str(raw_path)) as reader:
        for spectrum in reader:
            if spectrum.get(ms_level_key) != 1:
                continue

            mz = np.asarray(spectrum.get("m/z array", []))
            intensity = np.asarray(spectrum.get("intensity array", []))
            if mz.size == 0:
                continue

            rt_sec = scan_time_getter(spectrum)
            scans.append({"rt": rt_sec, "mz": mz, "intensity": intensity})
    return scans

def _mzml_scan_time(spectrum: dict) -> float:
    """Scan start time of an mzML spectrum in seconds (the file's unit decides whether to convert from minutes)."""
    scan_time = spectrum["scanList"]["scan"][0]["scan start time"]
    unit = getattr(scan_time, "unit_info", "minute")
    return float(scan_time) * 60.0 if unit == "minute" else float(scan_time)

def _mzxml_scan_time(spectrum: dict) -> float:
    """Retention time of an mzXML scan in seconds (converted from minutes when the file says so)."""
    scan_time = spectrum["retentionTime"]
    unit = getattr(scan_time, "unit_info", "minute")
    return float(scan_time) * 60.0 if unit == "minute" else float(scan_time)

def _spectrum_matches_precursor(spectrum: dict, target_mz: float, tolerance_da: float) -> bool:
    """
    True if any selected ion in this MS2 spectrum's precursor info falls
    within tolerance_da of target_mz.Handles both mzML's nested
    precursorList/selectedIonList structure and mzXML's flatter
    precursorMz list structure
    """
    if "precursorList" in spectrum:  # mzML shape
        for precursor in spectrum.get("precursorList", {}).get("precursor", []):
            for ion in precursor.get("selectedIonList", {}).get("selectedIon", []):
                mz = ion.get("selected ion m/z")
                if mz is not None and abs(float(mz) - target_mz) <= tolerance_da:
                    return True
        return False

    if "precursorMz" in spectrum:  # mzXML shape
        for precursor_entry in spectrum.get("precursorMz", []):
            mz = precursor_entry.get("precursorMz")
            if mz is not None and abs(float(mz) - target_mz) <= tolerance_da:
                return True
        return False

    return False

class PyteomicsSpectralPurityReader:
    """
    Precursor isolation purity: for each MS2 scan matching the
    target precursor, look at the preceding MS1 scan's peaks within the
    isolation window and compute what fraction belongs to the target ion.
    """

    TARGET_WINDOW_DA = 0.01

    def compute_precursor_purity(
            self, mzml_path: Path, precursor_mz: float, isolation_window_da: float
    ) -> Optional[float]:
        """
        Precursor purity: target signal / all signal inside the isolation window, measured in the MS1 scan immediately before each matching MS2 scan.

        Returns:
            A value between 0 and 1 (average over the MS2 scans of that precursor), or None if it cannot be computed
            (no MS2 scan for the precursor, or no preceding MS1 scan).

        Raises:
            ValueError: the file is neither .mzML nor .mzXML.
        """
        suffix = mzml_path.suffix.lower()
        if suffix == ".mzml":
            from pyteomics import mzml as reader_module
            level_key = "ms level"
        elif suffix == ".mzxml":
            from pyteomics import mzxml as reader_module
            level_key = "msLevel"
        else:
            raise ValueError(f"Unsupported raw file extension '{suffix}' for {mzml_path}")

        half_window = isolation_window_da / 2.0
        purities: list[float] = []
        last_ms1_peaks: Optional[np.ndarray] = None

        with reader_module.read(str(mzml_path)) as reader:
            for spectrum in reader:
                ms_level = spectrum.get(level_key)

                if ms_level == 1:
                    mz_arr = np.asarray(spectrum.get("m/z array", []))
                    intensity_arr = np.asarray(spectrum.get("intensity array", []))
                    last_ms1_peaks = (
                        np.column_stack([mz_arr, intensity_arr]) if mz_arr.size else None
                    )
                    continue

                if ms_level != 2 or last_ms1_peaks is None:
                    continue

                if not _spectrum_matches_precursor(spectrum, precursor_mz, 0.01):
                    continue

                mz_array = last_ms1_peaks[:, 0]
                intensity_array = last_ms1_peaks[:, 1]

                window_mask = (mz_array >= precursor_mz - half_window) & (
                        mz_array <= precursor_mz + half_window
                )
                window_total = intensity_array[window_mask].sum()
                if window_total <= 0:
                    continue

                target_mask = (mz_array >= precursor_mz - self.TARGET_WINDOW_DA) & (
                        mz_array <= precursor_mz + self.TARGET_WINDOW_DA
                )
                target_total = intensity_array[target_mask].sum()
                purities.append(float(target_total / window_total))

        if not purities:
            return None
        return sum(purities) / len(purities)

class PyteomicsMS2SpectrumReader:
    """Extracts the MS2 spectrum of a compound: the highest-TIC MS2 scan whose precursor matches, noise peaks removed."""
    def __init__(self, min_relative_intensity: float = 0.01):
        """`min_relative_intensity`: fragment peaks below this fraction of the spectrum's base peak are dropped."""
        self.min_relative_intensity = min_relative_intensity

    def get_ms2_spectrum(
        self, raw_path: Path, precursor_mz: float, precursor_tolerance_da: float = 0.01
    ) -> list[tuple[float, float]]:
        """
        Best MS2 spectrum for a precursor.

        Returns:
            [(mz, intensity), ...]; empty if no MS2 scan selected that precursor within the tolerance [Da].

        Raises:
            ValueError: the file is neither .mzML nor .mzXML.
        """
        suffix = raw_path.suffix.lower()
        if suffix == ".mzml":
            return self._get_ms2_spectrum_mzml(raw_path, precursor_mz, precursor_tolerance_da)
        elif suffix == ".mzxml":
            return self._get_ms2_spectrum_mzxml(raw_path, precursor_mz, precursor_tolerance_da)
        else:
            raise ValueError(f"Unsupported raw file extension '{suffix}' for {raw_path}")

    def _get_ms2_spectrum_mzml(self, mzml_path, precursor_mz, precursor_tolerance_da):
        """mzML implementation of get_ms2_spectrum: keep the matching MS2 scan with the highest total ion current."""
        from pyteomics import mzml
        best_peaks, best_tic = None, -1.0

        with mzml.read(str(mzml_path)) as reader:
            for spectrum in reader:
                if spectrum.get("ms level") != 2:
                    continue
                if not _spectrum_matches_precursor(spectrum, precursor_mz, precursor_tolerance_da):
                    continue
                mz_arr = np.asarray(spectrum.get("m/z array", []))
                intensity_arr = np.asarray(spectrum.get("intensity array", []))
                if mz_arr.size == 0:
                    continue
                peaks = np.column_stack([mz_arr, intensity_arr])
                tic = float(peaks[:, 1].sum())
                if tic > best_tic:
                    best_tic, best_peaks = tic, peaks

        return self._filter_peaks(best_peaks)

    def _get_ms2_spectrum_mzxml(self, mzxml_path, precursor_mz, precursor_tolerance_da):
        """mzXML implementation of get_ms2_spectrum."""
        from pyteomics import mzxml
        best_peaks, best_tic = None, -1.0

        with mzxml.read(str(mzxml_path)) as reader:
            for spectrum in reader:
                if spectrum.get("msLevel") != 2:
                    continue
                if not _spectrum_matches_precursor(spectrum, precursor_mz, precursor_tolerance_da):
                    continue
                mz_arr = np.asarray(spectrum.get("m/z array", []))
                intensity_arr = np.asarray(spectrum.get("intensity array", []))
                if mz_arr.size == 0:
                    continue
                peaks = np.column_stack([mz_arr, intensity_arr])
                tic = float(peaks[:, 1].sum())
                if tic > best_tic:
                    best_tic, best_peaks = tic, peaks

        return self._filter_peaks(best_peaks)

    def _filter_peaks(self, best_peaks):
        """Drop peaks below `min_relative_intensity` x base peak; returns a list of (mz, intensity) tuples."""
        if best_peaks is None:
            return []
        base_peak_intensity = float(best_peaks[:, 1].max())
        if base_peak_intensity <= 0:
            return []
        floor = base_peak_intensity * self.min_relative_intensity
        filtered = best_peaks[best_peaks[:, 1] >= floor]
        return [(float(mz), float(intensity)) for mz, intensity in filtered]

def compute_ms1_coelution_purity(
    ms1_scans: list[dict],
    target_mz: float,
    target_rt_sec: float,
    isolation_window_da: float,
    mz_ppm_tolerance: float = 10.0,
) -> float | None:
    """
    Computes precursor purity directly from MS1 co-elution data

    ms1_scans: output of load_ms1_scans() for one sample (list of
    {"rt": float_sec, "mz": ndarray, "intensity": ndarray}).

    Returns None if no MS1 scan exists near target_rt_sec (shouldn't
    normally happen if target_rt_sec came from a real detected feature
    in that sample) or if the target ion itself isn't found within the
    window.
    """
    if not ms1_scans:
        return None

    closest_scan = min(ms1_scans, key=lambda s: abs(s["rt"] - target_rt_sec))

    half_window = isolation_window_da / 2
    window_mask = (
        (closest_scan["mz"] >= target_mz - half_window)
        & (closest_scan["mz"] <= target_mz + half_window)
    )
    window_intensities = closest_scan["intensity"][window_mask]
    window_mzs = closest_scan["mz"][window_mask]

    if len(window_mzs) == 0:
        return None

    tol = target_mz * mz_ppm_tolerance / 1e6
    target_mask = abs(window_mzs - target_mz) <= tol
    target_intensity = window_intensities[target_mask].sum()

    total_window_intensity = window_intensities.sum()
    if total_window_intensity == 0:
        return None

    return float(target_intensity / total_window_intensity)

def _get_precursor_mz_mzml(spectrum: dict) -> float | None:
    """Selected-ion m/z of an mzML MS2 spectrum, or None."""
    for precursor in spectrum.get("precursorList", {}).get("precursor", []):
        for ion in precursor.get("selectedIonList", {}).get("selectedIon", []):
            mz = ion.get("selected ion m/z")
            if mz is not None:
                return float(mz)
    return None


def _get_precursor_mz_mzxml(spectrum: dict) -> float | None:
    """Precursor m/z of an mzXML MS2 scan, or None."""
    for precursor_entry in spectrum.get("precursorMz", []):
        mz = precursor_entry.get("precursorMz")
        if mz is not None:
            return float(mz)
    return None


def build_ms2_index(raw_path: Path) -> list[tuple[float, float, list[tuple[float, float]], float]]:
    """
    Parses a raw file ONCE, extracting every MS2 spectrum's precursor m/z,
    total ion current, peak list and retention time (seconds). Returns a list of
    (precursor_mz, tic, peaks, rt_sec), sorted by precursor_mz. Call once per
    sample and reuse -- avoids re-parsing the same file per group lookup.
    """
    suffix = raw_path.suffix.lower()
    entries = []

    if suffix == ".mzml":
        from pyteomics import mzml
        with mzml.read(str(raw_path)) as reader:
            for spectrum in reader:
                if spectrum.get("ms level") != 2:
                    continue
                precursor_mz = _get_precursor_mz_mzml(spectrum)
                if precursor_mz is None:
                    continue
                mz_arr = np.asarray(spectrum.get("m/z array", []))
                intensity_arr = np.asarray(spectrum.get("intensity array", []))
                if mz_arr.size == 0:
                    continue
                tic = float(intensity_arr.sum())
                entries.append((precursor_mz, tic, list(zip(mz_arr.tolist(), intensity_arr.tolist())),
                                _mzml_scan_time(spectrum)))

    elif suffix == ".mzxml":
        from pyteomics import mzxml
        with mzxml.read(str(raw_path)) as reader:
            for spectrum in reader:
                if spectrum.get("msLevel") != 2:
                    continue
                precursor_mz = _get_precursor_mz_mzxml(spectrum)
                if precursor_mz is None:
                    continue
                mz_arr = np.asarray(spectrum.get("m/z array", []))
                intensity_arr = np.asarray(spectrum.get("intensity array", []))
                if mz_arr.size == 0:
                    continue
                tic = float(intensity_arr.sum())
                entries.append((precursor_mz, tic, list(zip(mz_arr.tolist(), intensity_arr.tolist())),
                                _mzxml_scan_time(spectrum)))

    else:
        raise ValueError(f"Unsupported raw file extension '{suffix}' for {raw_path}")

    entries.sort(key=lambda e: e[0])
    return entries


def lookup_best_ms2(
    index: list,
    target_mz: float,
    tolerance_da: float,
    min_relative_intensity: float,
    target_rt_sec: float | None = None,
    rt_tolerance_sec: float | None = None,
) -> list[tuple[float, float]]:
    """
    In-memory lookup against a preloaded index (sorted by precursor m/z, so
    this is a bisect, not a scan of every spectrum).

    With target_rt_sec/rt_tolerance_sec set, only spectra triggered within that
    window of the feature's retention time qualify -- otherwise a spectrum from
    ANY chromatographic peak with that precursor m/z is used, which is how
    isomers get each other's MS2. Picks the highest-TIC remaining spectrum.
    """
    import bisect
    key = lambda e: e[0]
    lo = bisect.bisect_left(index, target_mz - tolerance_da, key=key)
    hi = bisect.bisect_right(index, target_mz + tolerance_da, key=key)
    candidates = index[lo:hi]
    if target_rt_sec is not None and rt_tolerance_sec is not None:
        candidates = [e for e in candidates if abs(e[3] - target_rt_sec) <= rt_tolerance_sec]
    if not candidates:
        return []
    best = max(candidates, key=lambda e: e[1])
    peaks_arr = np.array(best[2])
    base_peak = peaks_arr[:, 1].max()
    if base_peak <= 0:
        return []
    floor = base_peak * min_relative_intensity
    filtered = peaks_arr[peaks_arr[:, 1] >= floor]
    return [(float(mz), float(i)) for mz, i in filtered]
