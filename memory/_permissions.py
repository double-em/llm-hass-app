"""Filesystem permission diagnostics and best-effort fixes.

Background
----------
The HA supervisor mounts the addon's /data directory with mode 0755 owned
by root. The published image runs the Flask app as ``appuser`` (uid 1000),
so writes to /data/<subdir>/... fail with ``PermissionError``. The
permission-error path was previously swallowed by ``VectorStore._ensure_dirs``
which then silently fell back to an in-memory ChromaDB — recall returned
empty forever and there was no surface-level signal.

This module exposes three helpers:

* :func:`diagnose_data_dir` — read-only check of the data dir's mode/owner
  plus a probe of whether the *current process* can write into it.
* :func:`best_effort_fix_data_dir` — attempt `chmod` (and `chown` if
  allowed) to make the dir writable. Returns ``(ok, error_message)``.
* :func:`ensure_subdirs_writable` — for each known subdirectory
  (``voices``, ``persons``, ``samples``, ``memory``, ``sessions``,
  ``voiceprints``, ``enrollments``, ``vector_memory``), create it if
  missing and verify writability.

Used by :func:`app.load_memory_system` (before constructing
``VectorStore``) and surfaced via :func:`app.health` as the
``data_dir`` block. If the helpers fail, ``VectorStore`` will still
try to operate, but the failure is now visible in ``/api/health`` and
in the addon's startup log.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Subdirectories the addon needs writable. Keep this list in sync with the
# wrapper's run.sh and Dockerfile. ``vector_memory`` is added because
# VectorStore creates it lazily.
REQUIRED_SUBDIRS = (
    "voices",
    "persons",
    "samples",
    "memory",
    "sessions",
    "voiceprints",
    "enrollments",
    "vector_memory",
)


def _stat_safe(path: Path) -> Optional[os.stat_result]:
    try:
        return path.stat()
    except OSError:
        return None


def _mode_octal(path: Path) -> Optional[str]:
    st = _stat_safe(path)
    if st is None:
        return None
    return oct(stat.S_IMODE(st.st_mode))


def _uid_gid(path: Path) -> tuple[Optional[int], Optional[int]]:
    st = _stat_safe(path)
    if st is None:
        return None, None
    return st.st_uid, st.st_uid  # st_uid; st_gid (return tuple of 2 below)


def _owner(path: Path) -> dict:
    """Return {uid, gid} of the path's owner (or None for either on failure)."""
    st = _stat_safe(path)
    if st is None:
        return {"uid": None, "gid": None}
    return {"uid": st.st_uid, "gid": st.st_gid}


def _is_writable(path: Path) -> bool:
    """Can the *current* process write into ``path``?

    Uses ``os.access`` (effective uid check) which is the right test
    for "will my mkdir()/open() succeed", not just "is the bit set".
    """
    try:
        return os.access(str(path), os.W_OK | os.X_OK)
    except OSError:
        return False


def diagnose_data_dir(data_dir: str = "/data") -> dict:
    """Inspect ``data_dir`` and return a diagnostic dict.

    The returned dict is JSON-safe and is exposed verbatim via
    ``/api/health`` as ``data_dir``. Always returns a dict — never
    raises — so callers can serialize it without guarding.
    """
    path = Path(data_dir)
    info = {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir() if path.exists() else False,
        "mode": _mode_octal(path),
        "owner": _owner(path),
        "process_uid": os.getuid(),
        "process_euid": os.geteuid(),
        "process_gid": os.getgid(),
        "writable": False,
        "subdirs": {},
        "error": None,
    }

    try:
        if not info["exists"]:
            info["error"] = "data dir does not exist"
            return info
        if not info["is_dir"]:
            info["error"] = "data path exists but is not a directory"
            return info

        info["writable"] = _is_writable(path)

        for sub in REQUIRED_SUBDIRS:
            sp = path / sub
            sub_info = {
                "exists": sp.exists(),
                "mode": _mode_octal(sp),
                "owner": _owner(sp),
                "writable": False,
            }
            if sub_info["exists"]:
                sub_info["writable"] = _is_writable(sp)
            info["subdirs"][sub] = sub_info

    except Exception as e:  # pragma: no cover — defensive
        info["error"] = f"diagnose failed: {type(e).__name__}: {e}"
        logger.exception("diagnose_data_dir failed")

    return info


def best_effort_fix_data_dir(diag: dict) -> tuple[bool, Optional[str]]:
    """Try to make ``/data`` (and known subdirs) writable.

    Strategy:

    1. ``chmod 0o777`` on the data dir (HA supervisor forbids chown to
       arbitrary uids, but mode changes are usually allowed).
    2. For each required subdir, ``mkdir -p`` then ``chmod 0o777``.
    3. Re-diagnose. If still not writable, return ``(False, error)``.

    Returns ``(True, None)`` on success. ``(False, msg)`` on failure
    where ``msg`` is a short human-readable reason for the log.
    """
    path = Path(diag.get("path", "/data"))
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return False, f"could not create {path}: {e}"

    errs: list[str] = []
    try:
        # chmod first — works even if we don't own the dir (we usually don't).
        os.chmod(path, 0o777)
    except OSError as e:
        errs.append(f"chmod {path}: {e}")

    for sub in REQUIRED_SUBDIRS:
        sp = path / sub
        try:
            sp.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            errs.append(f"mkdir {sp}: {e}")
            continue
        try:
            os.chmod(sp, 0o777)
        except OSError as e:
            errs.append(f"chmod {sp}: {e}")

    # Try chown only if we have CAP_CHOWN (root). This is best-effort —
    # if it fails because the supervisor denies it, we fall back to 0777.
    if os.geteuid() == 0:
        try:
            import pwd  # local import — only root needs it

            try:
                pw = pwd.getpwnam("appuser")
                os.chown(path, pw.pw_uid, pw.pw_gid)
                for sub in REQUIRED_SUBDIRS:
                    sp = path / sub
                    if sp.exists():
                        os.chown(sp, pw.pw_uid, pw.pw_gid)
            except KeyError:
                errs.append("chown skipped: 'appuser' not in passwd db")
        except OSError as e:
            errs.append(f"chown {path}: {e}")

    # Re-diagnose to confirm.
    new_diag = diagnose_data_dir(str(path))
    if not new_diag["writable"]:
        return False, "; ".join(errs) or new_diag.get("error") or "still not writable"

    return True, None


def ensure_subdirs_writable(data_dir: str = "/data") -> dict:
    """Convenience wrapper: diagnose, fix if needed, return final diag.

    Returns the *post-fix* diagnostic dict. If the fix attempt fails,
    the returned ``writable: False`` + ``error: <reason>`` is the
    surface that ``/api/health`` exposes.
    """
    diag = diagnose_data_dir(data_dir)
    if diag["writable"]:
        return diag

    ok, err = best_effort_fix_data_dir(diag)
    diag = diagnose_data_dir(data_dir)  # refresh

    if not ok:
        diag["fix_error"] = err
        logger.error(
            "best_effort_fix_data_dir failed: %s. Addon will run with "
            "in-memory fallback for vector_memory; restart supervisor "
            "with /data mounted writable if this persists.",
            err,
        )
    else:
        logger.info("best_effort_fix_data_dir succeeded; /data is now writable")

    return diag


# Module-level cache: the diagnosis is cheap but `stat()` on every health
# call is wasteful. Refresh every 30 seconds; refresh-on-failure so we
# pick up fixes from outside (e.g. supervisor restart) promptly.
_last_check: dict = {"diag": None, "ts": 0.0, "ok": False}
_CACHE_TTL_SEC = 30.0

import time


def get_data_dir_status(data_dir: str = "/data") -> dict:
    """Cached wrapper around :func:`ensure_subdirs_writable`."""
    now = time.time()
    last = _last_check
    if last["diag"] is not None and (now - last["ts"]) < _CACHE_TTL_SEC and last["ok"]:
        return last["diag"]
    diag = ensure_subdirs_writable(data_dir)
    _last_check.update({"diag": diag, "ts": now, "ok": diag.get("writable", False)})
    return diag
