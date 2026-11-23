"""
Validation for names that end up in file paths (study ids, sample ids, uploaded file names).

Every one of them comes from a request body or a CSV, and Path("base") / name does NOT keep the result
inside "base": an absolute name replaces the base entirely ("/etc/x"), and ".." climbs out of it.
"""
from __future__ import annotations

import re
from pathlib import Path

# study ids also appear in URLs, DB rows, file names and page text: keep them plain
STUDY_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
_STUDY_ID_RE = re.compile(STUDY_ID_PATTERN)


def is_valid_study_id(value: str) -> bool:
    """True for a plain study id: 1-64 characters, letters/digits/`_`/`-`, starting with a letter or digit."""
    return isinstance(value, str) and bool(_STUDY_ID_RE.fullmatch(value))


def is_safe_name(value: str) -> bool:
    """
    One path component: non-empty, no separators or control characters, not hidden (so not "." / ".."),
    and short enough for a file name. Looser than a study id because sample ids are file-name stems.
    """
    if not isinstance(value, str) or not value or len(value) > 200:
        return False
    if value.startswith(".") or ".." in value:
        return False
    return not any(ch in value for ch in "/\\\x00") and not any(ord(ch) < 32 for ch in value)


def safe_join(base: Path, name: str) -> Path:
    """base / name, or ValueError if name is not a plain file name or the result leaves base."""
    if not is_safe_name(name):
        raise ValueError(f"unsafe file or folder name: {name!r}")
    base = Path(base).resolve()
    target = (base / name).resolve()
    if target.parent != base:
        raise ValueError(f"path escapes {base}: {name!r}")
    return target
