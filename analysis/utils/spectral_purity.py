"""Pass/fail classification of a precursor purity value."""
from typing import Optional

def classify_purity(purity: Optional[float], min_purity: float) -> str:
    """"unknown" if purity is None, "pass" if purity >= min_purity, otherwise "below_threshold"."""
    if purity is None:
        return "unknown"
    return "pass" if purity >= min_purity else "below_threshold"