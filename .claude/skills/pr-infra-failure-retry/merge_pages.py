"""Merge paginated gh API output.

gh api --paginate emits one JSON array per page on separate lines.
This script reads them all from stdin and emits a single flat JSON array.
"""
import json
import sys

pages = []
for line in sys.stdin:
    line = line.strip()
    if line:
        pages.extend(json.loads(line))
print(json.dumps(pages))
