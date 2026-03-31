"""Remove a file under /tmp/.

Usage: python3 rm_tmp.py <filepath>

Only operates on paths under /tmp/ for safety.
Silently succeeds if the file does not exist (like rm -f).
Exits 0 on success, 1 on error.
"""
import sys
import os
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: rm_tmp.py <filepath>", file=sys.stderr)
    sys.exit(1)

path = Path(sys.argv[1]).resolve()
# Accept /tmp, /private/tmp (macOS symlink), and $TMPDIR
_tmpdir = os.environ.get("TMPDIR", "")
_ALLOWED = ("/tmp/", "/private/tmp/") + ((str(Path(_tmpdir).resolve()) + "/",) if _tmpdir else ())
if not any(str(path).startswith(p) for p in _ALLOWED):
    print(f"Error: path must be under /tmp/ or $TMPDIR, got: {path}", file=sys.stderr)
    sys.exit(1)

try:
    path.unlink()
except FileNotFoundError:
    pass
