"""Check lock file age and acquire lock.

Usage: python3 check_tmp.py <filepath> [max_age_seconds]

If the lock file exists and is younger than max_age_seconds (default: 3600),
prints a message and exits 1 (caller should skip).
Otherwise, creates/touches the lock file and exits 0 (caller may proceed).

Only operates on paths under /tmp/ or $TMPDIR for safety.
"""
import sys
import os
import time
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: check_tmp.py <filepath> [max_age_seconds]", file=sys.stderr)
    sys.exit(1)

path = Path(sys.argv[1]).resolve()
max_age = int(sys.argv[2]) if len(sys.argv) >= 3 else 3600

_tmpdir = os.environ.get("TMPDIR", "")
_ALLOWED = ("/tmp/", "/private/tmp/") + ((str(Path(_tmpdir).resolve()) + "/",) if _tmpdir else ())
if not any(str(path).startswith(p) for p in _ALLOWED):
    print(f"Error: path must be under /tmp/ or $TMPDIR, got: {path}", file=sys.stderr)
    sys.exit(2)

if path.exists():
    lock_age = int(time.time() - path.stat().st_mtime)
    if lock_age < max_age:
        print(f"Previous cycle still running (lock age: {lock_age}s). Skipping.")
        sys.exit(1)

path.parent.mkdir(parents=True, exist_ok=True)
path.touch()
