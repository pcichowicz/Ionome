"""
Raw-file naming in ONE place: which extensions count as raw LC-MS files and how a sample id maps to a file.

Imports only core.safe_paths (itself dependency-free), so any stage (and any test) can use it without a database or settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.safe_paths import is_safe_name

# Order = preference when a sample has both formats side by side.
RAW_SUFFIXES: tuple[str, ...] = (".mzML", ".mzXML")
_LOWER = tuple(s.lower() for s in RAW_SUFFIXES)


def is_raw_file(path: Path | str) -> bool:
    """True if the file name ends in .mzML or .mzXML (any letter case)."""
    return Path(path).suffix.lower() in _LOWER


def find_raw_file(raw_dir: Path | str, sample_id: str) -> Optional[Path]:
    """<raw_dir>/<sample_id>.mzML or .mzXML (any letter case, mzML preferred); None if neither exists."""
    if not is_safe_name(str(sample_id)):
        return None  # never build a path from "../x" or "/abs/x"
    raw_dir = Path(raw_dir)
    for suffix in RAW_SUFFIXES:
        candidate = raw_dir / f"{sample_id}{suffix}"
        if candidate.is_file():
            return candidate
    if raw_dir.is_dir():  # .mzml / .MZML written by some converters
        wanted = {f"{sample_id}{s}".lower() for s in RAW_SUFFIXES}
        for p in sorted(raw_dir.iterdir()):
            if p.is_file() and p.name.lower() in wanted:
                return p
    return None


def list_raw_files(raw_dir: Path | str) -> list[Path]:
    """All raw files directly inside `raw_dir`, sorted by name; an empty list if the folder does not exist."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return []
    return sorted(p for p in raw_dir.iterdir() if p.is_file() and is_raw_file(p))
