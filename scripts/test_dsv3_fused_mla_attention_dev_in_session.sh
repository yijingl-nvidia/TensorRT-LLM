#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the DeepSeek-V3 / GLM-5 fused MLA context-attention development test
# inside the active gpu_dev_session from a login node.
#
# Examples:
#   scripts/test_dsv3_fused_mla_attention_dev_in_session.sh
#
#   scripts/test_dsv3_fused_mla_attention_dev_in_session.sh \
#     -q \
#     tests/unittest/_torch/models/test_modeling_deepseekv3_attention.py \
#     ::test_glm5_fp8_context_attention_matches_pytorch_reference \
#     -s -rs

set -euo pipefail

RUN_START_EPOCH="$(date +%s)"

format_duration() {
    local total="$1"
    local hours=$((total / 3600))
    local minutes=$(((total % 3600) / 60))
    local seconds=$((total % 60))

    if [ "$hours" -gt 0 ]; then
        printf '%dh %02dm %02ds' "$hours" "$minutes" "$seconds"
    elif [ "$minutes" -gt 0 ]; then
        printf '%dm %02ds' "$minutes" "$seconds"
    else
        printf '%ds' "$seconds"
    fi
}

python_quote() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\'/\\\'}"
    printf "'%s'" "$value"
}

join_python_words() {
    local out=""
    local word
    for word in "$@"; do
        out+="$(python_quote "$word"), "
    done
    printf '%s' "$out"
}

report_duration() {
    local status="$?"
    local end_epoch elapsed

    end_epoch="$(date +%s)"
    elapsed=$((end_epoch - RUN_START_EPOCH))

    echo ""
    echo "================================================================="
    if [ "$status" -eq 0 ]; then
        echo "  dsv3_fused_mla_attention_dev pytest OK"
    else
        echo "  dsv3_fused_mla_attention_dev pytest FAILED (exit $status)"
    fi
    echo "  Duration: $(format_duration "$elapsed")"
    echo "================================================================="
}

trap report_duration EXIT

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
DEFAULT_SRC="$(git -C "${HERE}/.." rev-parse --show-toplevel)"
SRC_DIR="${TRTLLM_SRC_DIR:-$DEFAULT_SRC}"
EXEC="${GPU_DEV_SESSION_EXEC:-${HOME}/dev/cluster-workspace/gpu_dev_session/exec.sh}"

[ -x "$EXEC" ] || {
    echo "ERROR: $EXEC not executable. Run gpu_dev_session/start.sh first." >&2
    exit 1
}

SESSION_NAME="${SESSION_NAME:-}"
if [ -n "$SESSION_NAME" ]; then
    SESSION_ENV_FILE="${HOME}/.cache/gpu_dev_session/${SESSION_NAME}.env"
else
    SESSION_ENV_FILE="${HOME}/.cache/gpu_dev_session/active.env"
fi
[ -f "$SESSION_ENV_FILE" ] || {
    echo "ERROR: no session env at $SESSION_ENV_FILE; run gpu_dev_session/start.sh first." >&2
    exit 1
}
# shellcheck source=/dev/null
source "$SESSION_ENV_FILE"

PYTEST_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --)
            shift
            PYTEST_ARGS+=("$@")
            break
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "${#PYTEST_ARGS[@]}" -eq 0 ]; then
    PYTEST_ARGS=(-q tests/unittest/_torch/models/test_modeling_deepseekv3_attention.py -s -rs)
fi

DEFAULT_OUT_BASE="/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_inference/users/${USER}"
OUT_ROOT=""
for OUT_BASE in "${STORAGE_DIR:-}" "$DEFAULT_OUT_BASE" "${TMPDIR:-/tmp}/trtllm_${USER}"; do
    [ -n "$OUT_BASE" ] || continue
    CANDIDATE_OUT_ROOT="${OUT_BASE}/nvbug6108841_logs"
    if mkdir -p "${CANDIDATE_OUT_ROOT}/trtllm" 2>/dev/null \
            && [ -w "${CANDIDATE_OUT_ROOT}/trtllm" ]; then
        OUT_ROOT="$CANDIDATE_OUT_ROOT"
        break
    fi
done
[ -n "$OUT_ROOT" ] || {
    echo "ERROR: no writable log directory found." >&2
    exit 1
}
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="fused_mla_attention_dev_test_${RUN_STAMP}${SESSION_NAME:+_session${SESSION_NAME}}"
RUN_DIR="${OUT_ROOT}/trtllm/${RUN_NAME}"
mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/test_dsv3_fused_mla_attention_dev.log"

SRC_DIR_PY="$(python_quote "$SRC_DIR")"
PYTEST_ARGS_PY="$(join_python_words "${PYTEST_ARGS[@]}")"

echo "================================================================="
echo "  dsv3_fused_mla_attention_dev pytest"
echo "================================================================="
echo "  session     : job ${GPU_DEV_JOB:-unknown} on ${GPU_DEV_NODE:-unknown}"
echo "  container   : ${GPU_DEV_CONTAINER:-unknown}"
echo "  src dir     : $SRC_DIR"
echo "  branch      : $(git -C "$SRC_DIR" branch --show-current 2>/dev/null)"
echo "  HEAD        : $(git -C "$SRC_DIR" log -1 --format='%h %s' 2>/dev/null)"
echo "  pytest args : ${PYTEST_ARGS[*]}"
echo "  log         : $LOG"
echo "================================================================="

INNER_PY="import os, subprocess, sys; os.chdir(${SRC_DIR_PY}); "
INNER_PY+="args = ['python3', '-m', 'pytest', ${PYTEST_ARGS_PY}]; "
INNER_PY+="print('Running: ' + ' '.join(args), flush=True); "
INNER_PY+="sys.exit(subprocess.call(args))"

set +e
"$EXEC" -- python3 -c "$INNER_PY" 2>&1 | tee "$LOG"
TEST_STATUS="${PIPESTATUS[0]}"
set -e

if [ "$TEST_STATUS" -ne 0 ]; then
    echo "" >&2
    echo "=================================================================" >&2
    echo "  dsv3_fused_mla_attention_dev pytest failed with exit code $TEST_STATUS" >&2
    echo "  Test log: $LOG" >&2
    echo "" >&2
    echo "  Tail of log:" >&2
    tail -120 "$LOG" >&2 || true
    echo "=================================================================" >&2
    exit "$TEST_STATUS"
fi

if grep -Eq "^[0-9]+ skipped" "$LOG" && ! grep -Eq "([0-9]+ failed|[0-9]+ passed)" "$LOG"; then
    echo "" >&2
    echo "=================================================================" >&2
    echo "  pytest completed but all selected tests were skipped" >&2
    echo "  Test log: $LOG" >&2
    echo "  Re-run output includes skip reasons because the default pytest args include -rs." >&2
    echo "=================================================================" >&2
    exit 2
fi

echo ""
echo "================================================================="
echo "  dsv3_fused_mla_attention_dev pytest complete"
echo "  Test log: $LOG"
echo "================================================================="
