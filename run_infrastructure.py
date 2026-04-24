"""run_infrastructure.py — Kaggle-safe run state, checkpoint, and logging.

This module makes the 4-6h IRP-LT + BiGAT pipeline
  (a) RESUMABLE after kernel death / Kaggle timeout,
  (b) quiet in the notebook (so __notebook__.ipynb does not balloon to 1 GB),
  (c) auditable via a single Results/run.log.

It is intentionally framework-agnostic beyond PyTorch (used for CheckpointManager)
and pandas (used for append_csv_unique).

Public API
----------
- RunLogger:            tee stdout/stderr to Results/run.log; keep notebook quiet
- RunState:             persist phase progress to Results/run_state.json
- CheckpointManager:    save/load last_model.pt + best_model.pt (model+optim+sched)
- append_csv_unique:    idempotent CSV append, de-dupes on key columns
- write_json_atomic:    atomic JSON writer (tmp + rename)
- quiet_subprocess:     run a child and stream stdout into the log file only
- safe_preview/json:    tiny notebook preview + full dump to log
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Run logger — tee stdout/stderr to Results/run.log
# ---------------------------------------------------------------------------

class _Tee(io.TextIOBase):
    """Write to multiple streams. Swallows per-stream failures."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


class RunLogger:
    """Mirror stdout/stderr to Results/run.log while keeping notebook output.

    Kaggle truncates `__notebook__.ipynb` when it exceeds a few hundred MB,
    but files under /kaggle/working survive. Writing to run.log gives us a
    durable record even if the notebook HTML gets nuked.
    """

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.results_dir / "run.log"
        self._file: Optional[io.TextIOBase] = None
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

    def activate(self) -> Path:
        if self._file is not None:
            return self.log_path
        # Append so resumed runs add to the same log history.
        self._file = open(self.log_path, "a", encoding="utf-8", buffering=1)
        self._file.write(f"\n===== RUN STARTED {datetime.now(timezone.utc).isoformat()} "
                         f"pid={os.getpid()} =====\n")
        sys.stdout = _Tee(self._orig_stdout, self._file)
        sys.stderr = _Tee(self._orig_stderr, self._file)
        return self.log_path

    def deactivate(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        if self._file is not None:
            self._file.close()
            self._file = None

    def log(self, msg: str) -> None:
        """File-only log line; does NOT print to the notebook."""
        if self._file is not None:
            self._file.write(f"[{datetime.now(timezone.utc).isoformat()}] {msg}\n")
            self._file.flush()


# ---------------------------------------------------------------------------
# 2. RunState — a tiny, resilient JSON state file
# ---------------------------------------------------------------------------

class RunState:
    """Track phase completion + arbitrary key/value metadata.

    Each phase has an 'ok' flag, start/finish timestamps, and whatever extra
    fields the caller passes in. Writes are atomic (tmp + rename) so a kernel
    death mid-write never corrupts the file.
    """

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.results_dir / "run_state.json"
        self.state: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                self.state = {}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, default=str)
        tmp.replace(self.path)

    def is_done(self, phase: str) -> bool:
        entry = self.state.get(phase) or {}
        return bool(entry.get("ok"))

    def mark_start(self, phase: str, **extra: Any) -> None:
        entry = dict(self.state.get(phase) or {})
        entry["started_at"] = datetime.now(timezone.utc).isoformat()
        entry["ok"] = False
        entry.update(extra)
        self.state[phase] = entry
        self._save()

    def mark_done(self, phase: str, **extra: Any) -> None:
        entry = dict(self.state.get(phase) or {})
        entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        entry["ok"] = True
        entry.update(extra)
        self.state[phase] = entry
        self._save()

    def update(self, phase: str, **extra: Any) -> None:
        entry = dict(self.state.get(phase) or {})
        entry.update(extra)
        self.state[phase] = entry
        self._save()

    def get(self, phase: str) -> Dict[str, Any]:
        return dict(self.state.get(phase) or {})


# ---------------------------------------------------------------------------
# 3. CheckpointManager — PyTorch last_model.pt + best_model.pt
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Save `last_model.pt` every epoch + `best_model.pt` on improvement.

    The caller controls the payload dict (so training loops can include
    optimizer, scheduler, epoch, best metric, feature-normalization stats,
    etc.). Writes are atomic — tmp file + rename — so a crash mid-save never
    leaves a corrupted .pt on disk.
    """

    def __init__(self, checkpoints_dir: Path):
        self.dir = Path(checkpoints_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.last_path = self.dir / "last_model.pt"
        self.best_path = self.dir / "best_model.pt"

    def _save(self, payload: Dict[str, Any], dst: Path) -> Path:
        import torch
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(dst)
        return dst

    def save_last(self, payload: Dict[str, Any]) -> Path:
        return self._save(payload, self.last_path)

    def save_best(self, payload: Dict[str, Any]) -> Path:
        return self._save(payload, self.best_path)

    def _load(self, path: Path, map_location: str) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        import torch
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    def load_last(self, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
        return self._load(self.last_path, map_location)

    def load_best(self, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
        return self._load(self.best_path, map_location)


# ---------------------------------------------------------------------------
# 4. Idempotent partial-output helpers
# ---------------------------------------------------------------------------

def append_csv_unique(
    path: Path,
    df: pd.DataFrame,
    key_cols: Sequence[str],
) -> int:
    """Append df to path, de-duped on key_cols. Returns the number of rows added.

    Contract:
      - If `path` does not exist, df is written verbatim.
      - If it exists, only rows whose key_cols tuple is NOT already present are
        appended (no rewrite of existing data).
      - Empty df → 0 rows added, no file touched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        return 0
    key_cols = [c for c in key_cols if c in df.columns]
    if not path.exists():
        df.to_csv(path, index=False)
        return len(df)
    if not key_cols:
        df.to_csv(path, mode="a", header=False, index=False)
        return len(df)
    try:
        existing = pd.read_csv(path, usecols=key_cols, dtype=str)
    except ValueError:
        # key_cols missing in existing file → just append without dedup.
        df.to_csv(path, mode="a", header=False, index=False)
        return len(df)
    existing_keys = set(map(tuple, existing.astype(str).to_numpy()))
    new_keys = df[key_cols].astype(str).apply(tuple, axis=1)
    mask = ~new_keys.isin(existing_keys)
    to_append = df.loc[mask.values]
    if to_append.empty:
        return 0
    to_append.to_csv(path, mode="a", header=False, index=False)
    return len(to_append)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Atomic JSON writer — tmp + rename. Safe against kernel death mid-write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 5. Subprocess streaming — never buffer GB of stdout into notebook HTML
# ---------------------------------------------------------------------------

def quiet_subprocess(
    cmd: Sequence[str],
    log_path: Path,
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    tag: Optional[str] = None,
    heartbeat_every: int = 500,
) -> int:
    """Run a subprocess; stream stdout+stderr line-by-line into log_path.

    Notebook receives only a start banner, a heartbeat every
    `heartbeat_every` lines (with elapsed seconds), and an end banner.
    The full stdout goes to the log file — which survives a kernel crash.
    Returns the child's exit code.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tag = tag or Path(cmd[1] if len(cmd) > 1 else cmd[0]).name
    print(f"[subprocess:{tag}] starting", flush=True)
    line_count = 0
    t0 = time.perf_counter()
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_f:
        log_f.write(f"\n===== [{tag}] {datetime.now(timezone.utc).isoformat()} =====\n")
        log_f.write("CMD: " + " ".join(str(c) for c in cmd) + "\n\n")
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_f.write(line)
            line_count += 1
            if heartbeat_every and (line_count % heartbeat_every == 0):
                print(f"  [{tag}] {line_count} lines, "
                      f"{time.perf_counter() - t0:.0f}s elapsed", flush=True)
        code = proc.wait()
        log_f.write(f"\n[exit] code={code} lines={line_count} "
                    f"seconds={time.perf_counter() - t0:.1f}\n")
    print(f"[subprocess:{tag}] done code={code} lines={line_count} "
          f"runtime={time.perf_counter() - t0:.1f}s", flush=True)
    return code


# ---------------------------------------------------------------------------
# 6. Tiny-preview helpers — keep DataFrames and dicts out of notebook HTML
# ---------------------------------------------------------------------------

def safe_preview(
    title: str,
    df: Optional[pd.DataFrame],
    *,
    rows: int = 5,
    log_path: Optional[Path] = None,
) -> None:
    """Print row/col counts + small head to notebook; full table to log."""
    if df is None or len(df) == 0:
        print(f"[{title}] empty", flush=True)
        return
    print(f"[{title}] rows={len(df)} cols={len(df.columns)}", flush=True)
    preview = df.head(rows).to_string(index=False)
    for line in preview.splitlines()[: rows + 1]:
        print("  " + line[:240], flush=True)
    if log_path is not None:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[{title}] FULL ({len(df)} rows)\n")
                f.write(df.to_string(index=False))
                f.write("\n")
        except Exception:
            pass


def safe_print_json(
    title: str,
    obj: Any,
    *,
    max_keys: int = 12,
    log_path: Optional[Path] = None,
) -> None:
    """Print a 1-line summary of a dict/list; full content goes to the log."""
    if isinstance(obj, dict):
        keys = list(obj.keys())
        shown = ", ".join(keys[:max_keys])
        suffix = "..." if len(keys) > max_keys else ""
        print(f"[{title}] type=dict keys={len(keys)} preview=[{shown}{suffix}]",
              flush=True)
    elif hasattr(obj, "__len__"):
        print(f"[{title}] type={type(obj).__name__} len={len(obj)}", flush=True)
    else:
        print(f"[{title}] type={type(obj).__name__}", flush=True)
    if log_path is not None:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[{title}]\n")
                json.dump(obj, f, indent=2, default=str)
                f.write("\n")
        except Exception:
            pass
