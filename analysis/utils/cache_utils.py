"""
Stage-level result cache: skip a stage when nothing it depends on has changed.

How it works:
  1. Each stage may define `cache_key(context)`: a hash of its own inputs (settings, file contents).
  2. The pipeline folds in two more things: the keys of all upstream stages (`context.stage_chain`) and a
     fingerprint of the stage's source code (`code_fingerprint`), so editing a stage or a helper it imports also
     invalidates it.
  3. The final key is compared with the checkpoint saved in `<results>/checkpoints/<stage>.json`. If it matches
     (and the stage's expected output files still exist) the stage is skipped and its saved qc_metrics are reused.

File hashes are remembered in `<results>/cache/filehash_cache.json` by (path, mtime, size) so large raw files are
not re-hashed on every run.
"""

from __future__ import annotations
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any

def _filehash_cache_path(cache_dir: Path) -> Path:
    """
    returns path to the 'filehash_cache.json' file
    """
    return cache_dir / "filehash_cache.json"

def load_filehash_cache(cache_dir: Path) -> dict[str, Any]:
    """
    Reads and loads 'filehash_cache.json' file, returns empty {} if does not exist
    """
    path = _filehash_cache_path(cache_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.decoder.JSONDecodeError, OSError):
        return {}

def save_filehash_cache(cache_dir: Path, cache: dict[str, Any]) -> None:
    """
    Writes contents to 'filehash_cache.json' file
    """
    _atomic_write(_filehash_cache_path(cache_dir), json.dumps(cache, indent=2))

def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + rename so a crash or a concurrent run never leaves half a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)

def hash_file(path: Path, cache: dict[str, Any]) -> str:
    """sha256 of file contents. Skips re-reading the file if its mtime and
    size match what's already in `cache` for this path.
    """
    key = str(path.resolve())
    stat = path.stat()
    signature = {"mtime": stat.st_mtime, "size": stat.st_size}

    cached = cache.get(key)
    if cached and cached.get("mtime") == signature["mtime"] and cached.get("size") == signature["size"]:
        return cached["sha256"]

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()

    cache[key] = {**signature, "sha256": sha256}
    return sha256

def hash_files(paths: list[Path], cache: dict[str, Any]) -> dict[str, str]:
    """`{file name: sha256}` for several files (sorted by path); uses and updates the same hash cache."""
    return {p.name: hash_file(p, cache) for p in sorted(paths)}

def compute_cache_key(payload: dict[str, Any]) -> str:
    """SHA-256 of the payload serialised as canonical JSON (sorted keys; non-JSON values via str())."""
    canonical = json.dumps(payload,sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def checkpoint_path(results_dir:Path, stage_name:str) -> Path:
    """
    The path to the checkpoint file, each stage has separate json file
    """
    return results_dir / "checkpoints" / f"{stage_name}.json"

def load_checkpoint(results_dir: Path, stage_name:str) -> dict[str,Any] | None:
    """
    Loads and reads the checkpoint json file for sha256 key
    """
    path = checkpoint_path(results_dir, stage_name)
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

def save_checkpoint(
        results_dir:Path,
        stage_name: str,
        key: str,
        qc_metrics_slice: Any,
        log_entry: dict[str, Any] | None
) -> None:
    """
    Saves and writes the checkpoint json file
    """
    path = checkpoint_path(results_dir, stage_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage_name,
        "key": key,
        "qc_metrics": qc_metrics_slice,
        "log_entry": log_entry,
    }
    _atomic_write(path, json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Code fingerprint: replaces "remember to bump CACHE_SCHEMA_VERSION"
# ---------------------------------------------------------------------------
_FINGERPRINT_PREFIX = "analysis.utils."
# modules whose changes cannot alter a stage's results
_FINGERPRINT_IGNORE = {"analysis.utils.cache_utils", "analysis.utils.protocols", "analysis.utils.timer",
                       "analysis.utils.schemas"}
_file_digest_memo: dict[tuple[str, float, int], str] = {}


def _digest_file(path: Path) -> str:
    """SHA-256 of a source file, memoised by (path, mtime, size) within the process."""
    st = path.stat()
    memo_key = (str(path), st.st_mtime, st.st_size)
    if memo_key not in _file_digest_memo:
        _file_digest_memo[memo_key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _file_digest_memo[memo_key]


def _module_path(name: str) -> Path | None:
    """Source file of an importable module, or None (not found, built-in, compiled)."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin)


def _imported_util_modules(path: Path) -> set[str]:
    """Names of the `analysis.utils.*` modules a source file imports (static scan; ignores the modules in _FINGERPRINT_IGNORE)."""
    names: set[str] = set()
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, OSError):
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.startswith(_FINGERPRINT_PREFIX):
                names.add(node.module)
            elif node.module == _FINGERPRINT_PREFIX.rstrip("."):
                names.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names if a.name.startswith(_FINGERPRINT_PREFIX))
    return {n for n in names if n not in _FINGERPRINT_IGNORE}


def code_fingerprint(stage: Any) -> str:
    """
    sha256 over the stage's own source file plus every analysis.utils module it imports,
    transitively. Editing any of them changes the stage's cache key, so stale results can't be
    reused after a code change. Stages may add modules (e.g. ones imported lazily inside a
    function) with a `code_modules = ("analysis.utils.x",)` class attribute.
    Not covered: third-party packages, and modules outside analysis.utils.
    """
    try:
        stage_file = Path(inspect.getsourcefile(type(stage)) or "")
    except TypeError:
        return "unknown"
    if not stage_file.exists():
        return "unknown"
    seen: dict[str, str] = {f"<stage>:{stage_file.name}": _digest_file(stage_file)}
    queue = list(_imported_util_modules(stage_file)) + list(getattr(stage, "code_modules", ()))
    visited: set[str] = set()
    while queue:
        name = queue.pop()
        if name in visited:
            continue
        visited.add(name)
        path = _module_path(name)
        if path is None:
            continue
        seen[name] = _digest_file(path)
        queue.extend(_imported_util_modules(path) - visited)
    return compute_cache_key(seen)


def missing_outputs(paths: list[Path]) -> list[Path]:
    """Paths that should exist for a cached stage result to still be valid, but don't."""
    return [Path(p) for p in paths if not Path(p).exists()]





