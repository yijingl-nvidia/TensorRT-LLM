"""Classify Jenkins testReport failures as infrastructure errors or real failures.

Reads the Jenkins testReport JSON from stdin.
Prints a summary line, then a JSON object {"infra": [...], "real": [...]}.
"""
import json
import sys

data = json.load(sys.stdin)
print(f'Summary: {data["passCount"]} passed, {data["failCount"]} failed, {data["skipCount"]} skipped')

infra = []
real = []

INFRA_DETAILS = {
    'test terminated unexpectedly',
    'stage run failed without result',
}

INFRA_KEYWORDS = [
    'test terminated unexpectedly',
    'stage run failed without result',
    'executor: lost connection',
    'node went offline',
    'out of memory',
    'oom killer',
    'killed',
]

for suite in data.get('suites', []):
    for case in suite.get('cases', []):
        if case.get('status') not in ('FAILED', 'REGRESSION'):
            continue
        name = f'{case["className"]}.{case["name"]}'
        details = (case.get('errorDetails') or '').strip()
        stdout = (case.get('stdout') or '').strip()
        stderr = (case.get('stderr') or '').strip()

        is_infra = (
            (not stdout and not stderr and (not details or details.lower() in INFRA_DETAILS))
            or any(kw in (details + stdout + stderr).lower() for kw in INFRA_KEYWORDS)
        )

        entry = {
            'name': name,
            'details': details[:300],
            'stdout_tail': stdout[-500:],
            'stderr_tail': stderr[-500:],
        }
        if is_infra:
            infra.append(entry)
        else:
            real.append(entry)

print(json.dumps({'infra': infra, 'real': real}, indent=2))
