"""
Storage backends for raw data. Only `LocalStorage` works today.

NOTE: `S3Storage` is an unfinished placeholder: `get_path` returns the bucket name rather than a path, and
`get_storage` reads `AWS_S3_REGION` while the settings define `AWS_REGION`. Do not set STORAGE_BACKEND=s3 until it is
implemented. Nothing in the pipeline calls this module yet; the stages read raw files through core.raw_files.
"""
import os
from abc import ABC, abstractmethod
from pathlib import Path

class StorageBackend(ABC):
    """Interface: map a key (relative file name) to a local path, store a file under a key, test for existence."""

    @abstractmethod
    def get_path(self, key: str) -> Path:
        """Local filesystem path for `key`."""
        ...
    @abstractmethod
    def put(self, local_path: Path, key: str) -> None:
        """Copy the file at `local_path` into the store under `key`."""
        ...
    @abstractmethod
    def exists(self, key: str) -> bool:
        """True if something is stored under `key`."""
        ...

class LocalStorage(StorageBackend):
    """Plain directory on disk; keys are paths relative to `root`."""

    def __init__(self, root: Path):
        self._root = root
    def get_path(self, key: str) -> Path:
        return self._root / key
    def put(self, local_path, key):
        destination = self.get_path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(local_path.read_bytes())
    def exists(self, key: str) -> bool:
        return self.get_path(key).exists()

class S3Storage(StorageBackend):
    """Placeholder for an S3 backend (see the module note); it currently behaves like a broken local store."""

    def __init__(self, bucket, region):
        self._bucket = bucket
        self._region = region
    def get_path(self, key: str) -> Path:
        return self._bucket
    def put(self, local_path, key):
        destination = self.get_path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(local_path.read_bytes())
    def exists(self, key: str) -> bool:
        return self.get_path(key).exists()


def get_storage() -> StorageBackend:
    """The backend chosen by STORAGE_BACKEND ("local" by default), rooted at LCMS_RAW_DATA_DIR."""
    backend = os.environ.get("STORAGE_BACKEND", "local")
    if backend == "s3":
        return S3Storage(bucket = os.environ["AWS_S3_BUCKET"], region = os.environ["AWS_S3_REGION"])
    return LocalStorage(root = os.environ["LCMS_RAW_DATA_DIR"])
