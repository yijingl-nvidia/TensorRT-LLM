"""Check whether /bot run was posted recently enough to skip a retry.

Usage: python3 check_recent_bot_run.py <ISO8601_timestamp_or_null>

Exits 0 and prints SKIP if the timestamp is within the last 30 minutes.
Exits 0 and prints OK otherwise (missing, null, or old timestamp).
"""
import sys
from datetime import datetime, timezone, timedelta

ts = sys.argv[1] if len(sys.argv) > 1 else ''
if ts and ts != 'null':
    last = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    age = datetime.now(timezone.utc) - last
    print(f'Last /bot run: {ts} ({int(age.total_seconds() / 60)} min ago)', file=sys.stderr)
    if age < timedelta(minutes=30):
        print('SKIP')
        sys.exit(0)
print('OK')
