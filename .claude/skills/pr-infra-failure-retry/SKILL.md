---
name: pr-infra-failure-retry
description: Monitor open TensorRT-LLM PRs, detect CI failures, classify them as infrastructure errors vs real failures, and auto-retry infra errors by posting /bot run. Invoke via /loop for continuous monitoring, or manually to check all open PRs once.
---

# PR Infra Failure Retry

## Overview

Monitor all open non-draft PRs authored by the current user, detect CI failures, diagnose each failure via the Jenkins testReport API, and take action:
- **Infrastructure failure** (killed process, empty output, timeout, OOM): automatically retry by posting `/bot run`.
- **Real failure** (assertion error, import error, logic bug, etc.): inform the user — no local GPU or Docker available on this machine to fix it.

## Environment

- **GitHub repo:** `NVIDIA/TensorRT-LLM`
- **CI bot user:** `tensorrt-cicd`
- **Jenkins shortcut:** `https://nv/trt-llm-cicd` (requires corporate network)
- **PR author filter:** resolved dynamically from `gh auth` — see Step 0
- **Trigger command:** post `/bot run` as an issue comment on the PR

## Required GitHub Authentication

All GitHub operations use the `gh` CLI. Authenticate once with `gh auth login` before running this skill.

### Minimum PAT permissions

**Fine-grained PAT (recommended — least privilege):**

| Permission | Level | Used for |
|---|---|---|
| Metadata | Read | Required by GitHub for all fine-grained PATs |
| Pull requests | Read | List open PRs, read PR head SHA |
| Commit statuses | Read | Check CI pass/fail state |
| Issues | Read and Write | Read CI bot comments; post `/bot run` |

Scope the PAT to the `NVIDIA/TensorRT-LLM` repository only.

**Classic PAT (alternative):**

| Scope | Used for |
|---|---|
| `public_repo` | All read + write operations on this public repo |
| `read:user` | Resolve authenticated username via `gh api user` |

## Workflow

Each invocation runs one full cycle. Use with `/loop` for continuous monitoring (e.g. `/loop 30m /pr-infra-failure-retry`).

### Step 0: Lock guard + resolve current user

```bash
LOCK_FILE="${TMPDIR:-/tmp}/pr_infra_retry.lock"
SKILL_DIR="$(git rev-parse --show-toplevel)/.claude/skills/pr-infra-failure-retry"

if [ -f "$LOCK_FILE" ]; then
  lock_age=$(( $(date +%s) - $(date -r "$LOCK_FILE" +%s 2>/dev/null || echo 0) ))
  if [ "$lock_age" -lt 3600 ]; then
    echo "Previous cycle still running (lock age: ${lock_age}s). Skipping."
    exit 0
  fi
fi
python3 "$SKILL_DIR/touch_tmp.py" "$LOCK_FILE"

GH_USER=$(gh api user --jq .login)
echo "Monitoring PRs for: $GH_USER"
```

Remove the lock at the end of the cycle (or on early exit):
```bash
python3 "$SKILL_DIR/rm_tmp.py" "$LOCK_FILE"
```

### Step 1: Survey all open PRs

> **Important:** This repo has many open PRs (100+). Use `--paginate` to fetch all pages; without it, older PRs are silently missed. `gh --paginate` emits one JSON array per page — merge them with `merge_pages.py`.

```bash
SKILL_DIR="$(git rev-parse --show-toplevel)/.claude/skills/pr-infra-failure-retry"

PRS=$(gh api --paginate "repos/NVIDIA/TensorRT-LLM/pulls?state=open&per_page=100" \
  --jq '[.[] | select(.draft == false) | select(.user.login == "'"$GH_USER"'") |
    {number: .number, title: .title[:60], branch: .head.ref, sha: .head.sha}]' \
  | python3 "$SKILL_DIR/merge_pages.py")
echo "$PRS"
```

This returns a JSON array of `{number, title, branch, sha}`. Iterate over each in subsequent steps.

### Step 2: Get CI status for each PR

For each PR, use the head SHA to query the combined commit status:

```bash
SHA=<head_sha>
gh api "repos/NVIDIA/TensorRT-LLM/commits/${SHA}/status" \
  --jq '{state: .state, description: (.statuses[0].description // "")}'
```

| CI state    | Action                    |
|-------------|---------------------------|
| `pending`   | Skip — wait for it        |
| `success`   | Skip — nothing to do      |
| `failure`   | Proceed to Step 3         |
| no statuses | Skip — CI not yet started |

### Step 3: Get Jenkins build number

The `tensorrt-cicd` bot posts an issue comment containing the Jenkins build URL. Extract the build number from the most recent such comment:

```bash
PR_NUM=<pr_number>
BUILD_NUM=$(gh api "repos/NVIDIA/TensorRT-LLM/issues/${PR_NUM}/comments" --jq \
  '[.[] | select(.user.login == "tensorrt-cicd") | select(.body | test("L0_MergeRequest_PR"))] | last | .body' \
  | grep -oE 'L0_MergeRequest_PR/[0-9]+' | grep -oE '[0-9]+$')
echo "BUILD_NUM=$BUILD_NUM"
```

If `BUILD_NUM` is empty, skip the PR (CI may have been manually cancelled or the bot has not yet commented).

### Step 4: Query Jenkins testReport and classify failures

The Jenkins server is `prod.blsm.nvidia.com` (accessible via `*.nvidia.com` sandbox allowlist). The `https://nv/trt-llm-cicd` shortcut used in bot comments redirects there, but is blocked by the Claude Code sandbox — use the real hostname directly:

```bash
JENKINS_BASE="https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/main/job/L0_MergeRequest_PR"
```

Verify reachability:

```bash
curl -sf "${JENKINS_BASE}/api/json?tree=displayName" -o /dev/null
if [ $? -ne 0 ]; then
  echo "ERROR: cannot reach Jenkins at $JENKINS_BASE. Skipping Jenkins steps."
  # mark PR as ERROR in report
fi
```

If unreachable, mark the PR as `ERROR` in the report.

Fetch and classify all failures:

```bash
curl -s "${JENKINS_BASE}/${BUILD_NUM}/testReport/api/json" \
  | python3 "$SKILL_DIR/classify_failures.py"
```

### Step 5: Take action based on classification

#### All failures are infrastructure errors → retry CI

First check whether `/bot run` was already posted within the last 30 minutes to avoid duplicate triggers:

```bash
LAST_RUN=$(gh api "repos/NVIDIA/TensorRT-LLM/issues/${PR_NUM}/comments" --jq \
  '[.[] | select(.user.login == "'"$GH_USER"'") | select(.body == "/bot run") | .created_at] | last')

# Skip if posted within last 30 minutes
python3 "$SKILL_DIR/check_recent_bot_run.py" "$LAST_RUN"
```

If not recently posted, post `/bot run`:

```bash
gh api "repos/NVIDIA/TensorRT-LLM/issues/${PR_NUM}/comments" \
  --method POST \
  --field body="/bot run"
```

Log: `PR #<num>: all <N> failures are infra → posted /bot run`

#### Any real failures → inform the user, do NOT retry

Print a clear summary for each real failure:

```
PR #<num> has <N> real failure(s) that need human attention:

  - <className>.<testName>
    Error: <errorDetails>
    Stdout tail: <last 500 chars>
    Stderr tail: <last 500 chars>

  Action required: fix the code before retrying CI.
  No local GPU/Docker available on this machine to auto-fix.
```

Do NOT post `/bot run`. Retrying with known real failures wastes CI capacity.

#### Mixed (infra + real) → report all, do NOT retry

Same as above — report everything and ask the user to fix the real failures first.

### Step 6: Report

Print a summary table covering **every** open non-draft PR:

```
=== PR Infra Retry Cycle ===
PR      Action       Branch                          Details
#12132  RETRIED      fix_prometheus_cache_metrics    2 infra failures → /bot run posted
#12099  WAITING      feature/kv-cache-metrics        pipeline pending
#12050  OK           fix/attention-bug               CI green
#11980  NEEDS FIX    refactor/moe-routing            1 real failure: test_moe_routing.test_top2 AssertionError

Actions: 1 retried, 1 needs fix, 1 waiting, 1 ok
```

Action labels:
- `OK` — CI green, nothing to do
- `WAITING` — CI pending/running, skipped
- `RETRIED` — all failures were infra, `/bot run` posted
- `NEEDS FIX` — one or more real failures, user must act
- `SKIPPED` — no Jenkins build found or CI not started
- `ERROR` — failed to reach Jenkins (no corporate network)

## Infrastructure Failure Heuristics

| Pattern | Verdict |
|---------|---------|
| `"Test terminated unexpectedly"` + empty stdout + empty stderr | Infra — process killed by runner |
| `"Stage run failed without result"` + empty stdout + empty stderr | Infra — Jenkins stage crashed before producing results |
| Empty `errorDetails`, empty `stdout`, empty `stderr` | Infra — silent kill |
| `"executor: lost connection"` or `"node went offline"` | Infra — agent crash |
| `"Out of memory"` / `"OOM killer"` / `"Killed"` in output | Infra — OOM on test node |
| Python traceback, `AssertionError`, `ImportError` | **Real failure** |
| Build error (`ninja: build stopped`, `error:`, `undefined reference`) | **Real failure** |
| Timeout with partial output showing test progression | Likely real (slow test) — report to user |

## Troubleshooting

### Jenkins unreachable / `blocked-by-allowlist`

The `https://nv/trt-llm-cicd` shortcut used in bot PR comments is **blocked by the Claude Code sandbox** — `nv` is not in the allowed hosts list. The skill uses `prod.blsm.nvidia.com` directly, which is covered by the `*.nvidia.com` sandbox allowlist.

If you see `blocked-by-allowlist` errors, verify the skill is using `prod.blsm.nvidia.com` and not the `nv` shortcut.

### TLS certificate errors (`x509: OSStatus -26276` or similar)

This is a **Claude Code sandbox restriction**, not a real certificate problem. `gh` uses Go's TLS stack, which calls the macOS `Security.framework` (`com.apple.trustd.agent`) to verify certificates. The fix is `sandbox.enableWeakerNetworkIsolation: true` in `~/.claude/settings.json` — this grants the sandbox access to `trustd` so Go-based tools can verify TLS without disabling the sandbox entirely.

If you see this error and the setting is already in place, check that `~/.claude/settings.json` was reloaded (restart Claude Code or open `/hooks`).

### Lock file permission error (`touch: Operation not permitted`)

The lock file uses `$TMPDIR` (e.g. `/var/folders/.../T/`) which is always sandbox-writable. If you see this error, check that `$TMPDIR` is set in your environment (`echo $TMPDIR`). It should be non-empty on macOS and most Linux systems.

## Important Rules

- **Never hardcode PR numbers.** Discover all open non-draft PRs dynamically every cycle.
- **Never retry a PR with real failures** — report them and stop.
- **No GPU / Docker on this machine.** When a real failure is found, describe it clearly and stop.
- **Corporate network required** for Jenkins. If unreachable, report `ERROR` and skip Jenkins steps.
