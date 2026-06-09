#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the DeepSeek-V3 fused MLA development dump tests inside the active
# gpu_dev_session from a login node.
#
# Examples:
#   scripts/test_dsv3_fused_mla_dev_in_session.sh
#
#   scripts/test_dsv3_fused_mla_dev_in_session.sh \
#     -q \
#     tests/unittest/_torch/models/test_modeling_deepseekv3_fused_mla.py \
#     ::test_deepseekv3_fused_mla_dump_projection_smoke[0] \
#     -s -rs
#
#   TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_MAX_NUM_TOKENS=all \
#     scripts/test_dsv3_fused_mla_dev_in_session.sh

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

shell_quote() {
    printf '%q' "$1"
}

expand_leading_tilde() {
    local path="$1"
    case "$path" in
        "~")
            printf '%s\n' "$HOME"
            ;;
        "~/"*)
            printf '%s/%s\n' "$HOME" "${path#~/}"
            ;;
        *)
            printf '%s\n' "$path"
            ;;
    esac
}

join_shell_words() {
    local out=""
    local word
    for word in "$@"; do
        out+="$(shell_quote "$word") "
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
        echo "  dsv3_fused_mla_dev pytest OK"
    else
        echo "  dsv3_fused_mla_dev pytest FAILED (exit $status)"
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
    PYTEST_ARGS=(-q tests/unittest/_torch/models/test_modeling_deepseekv3_fused_mla.py -s -rs)
fi

DEFAULT_DEBUG_OUTPUT_DIR="${HOME}/dev/mla-debug-output"
SRC_SIBLING_DEBUG_OUTPUT_DIR="${SRC_DIR}/../mla-debug-output"
if compgen -G "${SRC_SIBLING_DEBUG_OUTPUT_DIR}/r*_l*_*.pt" >/dev/null; then
    DEFAULT_DEBUG_OUTPUT_DIR="$SRC_SIBLING_DEBUG_OUTPUT_DIR"
fi
DEBUG_OUTPUT_DIR_RAW="${TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_DEBUG_OUTPUT_DIR:-}"
if [ -z "$DEBUG_OUTPUT_DIR_RAW" ]; then
    DEBUG_OUTPUT_DIR_RAW="${TRTLLM_DEEPSEEKV3_FUSED_MLA_DEBUG_OUTPUT_DIR:-}"
fi
if [ -z "$DEBUG_OUTPUT_DIR_RAW" ]; then
    DEBUG_OUTPUT_DIR_RAW="$DEFAULT_DEBUG_OUTPUT_DIR"
fi
DEBUG_OUTPUT_DIR_VALUE="$(expand_leading_tilde "$DEBUG_OUTPUT_DIR_RAW")"
MAX_TOKENS_VALUE="${TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_MAX_NUM_TOKENS:-128}"
EXTRA_OP_LIBRARY_VALUE="${TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_EXTRA_OP_LIBRARY:-}"

DEFAULT_OUT_ROOT="/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_inference/users/${USER}"
OUT_ROOT="${STORAGE_DIR:-$DEFAULT_OUT_ROOT}/nvbug6108841_logs"
RUN_DIR="${OUT_ROOT}/trtllm/fused_mla_dev_test_$(date +%Y%m%d_%H%M%S)${SESSION_NAME:+_session${SESSION_NAME}}"
mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/test_dsv3_fused_mla_dev.log"

PYTEST_ARGS_Q="$(join_shell_words "${PYTEST_ARGS[@]}")"
SRC_DIR_Q="$(shell_quote "$SRC_DIR")"
DEBUG_OUTPUT_DIR_Q="$(shell_quote "$DEBUG_OUTPUT_DIR_VALUE")"
MAX_TOKENS_Q="$(shell_quote "$MAX_TOKENS_VALUE")"
EXTRA_OP_LIBRARY_Q="$(shell_quote "$EXTRA_OP_LIBRARY_VALUE")"

echo "================================================================="
echo "  dsv3_fused_mla_dev pytest"
echo "================================================================="
echo "  session       : job ${GPU_DEV_JOB:-unknown} on ${GPU_DEV_NODE:-unknown}"
echo "  container     : ${GPU_DEV_CONTAINER:-unknown}"
echo "  src dir       : $SRC_DIR"
echo "  branch        : $(git -C "$SRC_DIR" branch --show-current 2>/dev/null)"
echo "  HEAD          : $(git -C "$SRC_DIR" log -1 --format='%h %s' 2>/dev/null)"
echo "  debug output  : $DEBUG_OUTPUT_DIR_VALUE"
echo "  max tokens    : $MAX_TOKENS_VALUE"
if [ -n "$EXTRA_OP_LIBRARY_VALUE" ]; then
    echo "  extra op lib  : $EXTRA_OP_LIBRARY_VALUE"
fi
echo "  pytest args   : ${PYTEST_ARGS[*]}"
echo "  log           : $LOG"
echo "================================================================="

INNER_CMD="
    set -euo pipefail
    cd ${SRC_DIR_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_DEBUG_OUTPUT_DIR=${DEBUG_OUTPUT_DIR_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_MAX_NUM_TOKENS=${MAX_TOKENS_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_EXTRA_OP_LIBRARY=${EXTRA_OP_LIBRARY_Q}
    pytest ${PYTEST_ARGS_Q}
"

set +e
"$EXEC" -- bash -c "$INNER_CMD" 2>&1 | tee "$LOG"
TEST_STATUS="${PIPESTATUS[0]}"
set -e

if [ "$TEST_STATUS" -ne 0 ]; then
    echo "" >&2
    echo "=================================================================" >&2
    echo "  dsv3_fused_mla_dev pytest failed with exit code $TEST_STATUS" >&2
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
echo "  dsv3_fused_mla_dev pytest complete"
echo "  Log: $LOG"
echo "================================================================="
