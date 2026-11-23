"""Matching a sample feature against blank (background) features."""
from typing import Optional

def is_blank_match(
    feature: dict,
    blank_features: list[dict],
    mz_tolerance_ppm: float,
    rt_tolerance_sec: float,
) -> Optional[dict]:
    """
    Return the first blank feature that matches `feature` within
    tolerance, or None.

    Both must hold: m/z difference <= mz_tolerance_ppm and retention-time difference <= rt_tolerance_sec.
    """
    for blank in blank_features:
        mz_ppm_error = abs(feature["mz"] - blank["mz"]) / blank["mz"] * 1e6
        rt_diff = abs(feature["rt"] - blank["rt"])
        if mz_ppm_error <= mz_tolerance_ppm and rt_diff <= rt_tolerance_sec:
            return blank
    return None