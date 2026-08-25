"""
On-demand local cache for HDF5 files stored on Google Drive / OneDrive.

Google Drive FUSE cannot reliably serve HDF5 random reads. When the data_dir
points at a cloud mount, copy each fire file to local disk on first use and
reuse it for the rest of the Colab session.

Set HDF5_CACHE_DIR to override the cache location.
Set HDF5_CACHE_DISABLE=1 to turn caching off.
Set HDF5_CACHE_FORCE=1 to always cache, even for local paths.
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import h5py

# Colab's /content is local SSD and is the fast path there; elsewhere fall back
# to the platform temp dir so this module has no Linux-only assumptions.
DEFAULT_CACHE_DIR = (
    "/content/.hdf5_cache"
    if os.path.isdir("/content")
    else os.path.join(tempfile.gettempdir(), "hdf5_cache")
)
INDEX_FILENAME = "dataset_index.json"


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def should_use_cache(path: str) -> bool:
    if os.environ.get("HDF5_CACHE_DISABLE", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("HDF5_CACHE_FORCE", "").lower() in ("1", "true", "yes"):
        return True
    norm = _normalize(path).lower()
    return "/drive/" in norm or "onedrive" in norm or "google drive" in norm


def cache_dir() -> str:
    return os.environ.get("HDF5_CACHE_DIR", DEFAULT_CACHE_DIR)


def _cache_path(src_path: str) -> str:
    norm = _normalize(os.path.abspath(src_path))
    root = _normalize(os.path.abspath(cache_dir()))
    if norm.startswith(root):
        return src_path

    marker = "/MyDrive/"
    if marker in norm:
        rel = norm.split(marker, 1)[1]
    else:
        rel = os.path.basename(norm)
    return os.path.join(cache_dir(), rel)


def cached_hdf5_path(src_path: str, retries: int = 5) -> str:
    """Return a local path for reading an HDF5 file, copying from cloud if needed.

    Safe for concurrent DataLoader workers: each worker copies to a unique temp
    file and then atomically renames it into place. This avoids the race where
    one worker reads (or removes) a destination file that another worker is
    still mid-write, which surfaced as h5py "file signature not found" errors.
    """
    if not should_use_cache(src_path):
        return src_path

    dst = _cache_path(src_path)

    try:
        src_size = os.path.getsize(src_path)
    except OSError:
        # Can't stat the source; let the caller's open() raise a clear error.
        return src_path

    if os.path.isfile(dst) and os.path.getsize(dst) == src_size:
        return dst

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    last_err = None
    for attempt in range(retries):
        # Another worker may have finished the copy while we were waiting.
        try:
            if os.path.isfile(dst) and os.path.getsize(dst) == src_size:
                return dst
        except OSError:
            pass

        # Unique per-process/per-attempt temp path so concurrent workers never
        # write to the same file. os.replace into dst is atomic on the same
        # filesystem, so readers only ever see a complete file.
        tmp = f"{dst}.tmp.{os.getpid()}.{attempt}"
        try:
            shutil.copy2(src_path, tmp)
            if os.path.getsize(tmp) == src_size:
                os.replace(tmp, dst)
                return dst
        except OSError as err:
            last_err = err
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

        time.sleep(min(2 ** attempt, 30))

    raise OSError(
        f"Failed to cache HDF5 from cloud storage after {retries} attempts: {src_path}"
    ) from last_err


def index_path(data_dir: str) -> str:
    return os.path.join(data_dir, INDEX_FILENAME)


def load_hdf5_index(data_dir: str) -> Optional[dict]:
    path = index_path(data_dir)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_hdf5_index(data_dir: str, index: dict) -> None:
    path = index_path(data_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def n_timesteps_from_index(index: dict, year: int, fire_name: str) -> Optional[int]:
    year_entry = index.get(str(year), {})
    fire_entry = year_entry.get(fire_name)
    if fire_entry is None:
        return None
    return int(fire_entry["n_timesteps"])


def read_n_timesteps(hdf5_path: str) -> int:
    local_path = cached_hdf5_path(hdf5_path)
    with h5py.File(local_path, "r") as f:
        return len(f["data"])
