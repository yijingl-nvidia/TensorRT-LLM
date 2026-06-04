#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the DeepSeek-V3 fused expert-up development Torch op tests inside the
# active gpu_dev_session.
#
# Default dev op:
#   cpp/build/dsv3_fused_expert_up_dev/libdsv3_fused_expert_up_dev.so
#   torch.ops.trtllm.dsv3_fused_expert_up_dev
#
# Examples:
#   scripts/test_dsv3_fused_expert_up_dev_in_session.sh
#
#   TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PHASE=profile \
#     scripts/test_dsv3_fused_expert_up_dev_in_session.sh
#
#   scripts/test_dsv3_fused_expert_up_dev_in_session.sh \
#     -q tests/unittest/_torch/models/test_modeling_deepseekv3_fused_moe.py::test_deepseekv3_fused_moe_profile_phase -s

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
        echo "  dsv3_fused_expert_up_dev pytest OK"
    else
        echo "  dsv3_fused_expert_up_dev pytest FAILED (exit $status)"
    fi
    echo "  Test duration: $(format_duration "$elapsed")"
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

BUILD_DIR="${DSV3_FUSED_EXPERT_UP_DEV_BUILD_DIR:-${SRC_DIR}/cpp/build/dsv3_fused_expert_up_dev}"
OUT_SO_DEFAULT="${DSV3_FUSED_EXPERT_UP_DEV_OUTPUT_SO:-${BUILD_DIR}/libdsv3_fused_expert_up_dev.so}"
EXTRA_OP_LIBRARY_VALUE="${TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_EXTRA_OP_LIBRARY:-$OUT_SO_DEFAULT}"
FUSED_EXPERT_UP_OP_VALUE="${TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_FUSED_EXPERT_UP_OP:-dsv3_fused_expert_up_dev}"
TEST_PHASE_VALUE="${TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PHASE:-both}"
MAX_TOKENS_VALUE="${TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_MAX_FUSED_KERNEL_NUM_TOKENS:-4}"
PROFILE_WARMUP_VALUE="${TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PROFILE_WARMUP_ITERS:-20}"
PROFILE_ITERS_VALUE="${TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PROFILE_ITERS:-100}"
DEBUG_OUTPUT_DIR_VALUE="$(expand_leading_tilde \
    "${TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_DEBUG_OUTPUT_DIR:-${TRTLLM_DEEPSEEKV3_FUSED_MOE_DEBUG_OUTPUT_DIR:-${HOME}/dev/debug_output}}")"
PROFILE_SUMMARY="${DEBUG_OUTPUT_DIR_VALUE}/fused_moe_profile_times.txt"
OUT_ROOT="${STORAGE_DIR:-/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_inference/users/${USER}}/nvbug6108841_logs"
RUN_DIR="${OUT_ROOT}/trtllm/fused_up_dev_test_$(date +%Y%m%d_%H%M%S)${SESSION_NAME:+_session${SESSION_NAME}}"
mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/test_dsv3_fused_expert_up_dev.log"

if [ "$#" -eq 0 ]; then
    PYTEST_ARGS=(-q tests/unittest/_torch/models/test_modeling_deepseekv3_fused_moe.py -s)
else
    PYTEST_ARGS=("$@")
fi

PYTEST_ARGS_Q="$(join_shell_words "${PYTEST_ARGS[@]}")"
SRC_DIR_Q="$(shell_quote "$SRC_DIR")"
EXTRA_OP_LIBRARY_Q="$(shell_quote "$EXTRA_OP_LIBRARY_VALUE")"
FUSED_EXPERT_UP_OP_Q="$(shell_quote "$FUSED_EXPERT_UP_OP_VALUE")"
TEST_PHASE_Q="$(shell_quote "$TEST_PHASE_VALUE")"
MAX_TOKENS_Q="$(shell_quote "$MAX_TOKENS_VALUE")"
PROFILE_WARMUP_Q="$(shell_quote "$PROFILE_WARMUP_VALUE")"
PROFILE_ITERS_Q="$(shell_quote "$PROFILE_ITERS_VALUE")"
DEBUG_OUTPUT_DIR_Q="$(shell_quote "$DEBUG_OUTPUT_DIR_VALUE")"

echo "================================================================="
echo "  dsv3_fused_expert_up_dev pytest"
echo "================================================================="
echo "  session       : job ${GPU_DEV_JOB:-unknown} on ${GPU_DEV_NODE:-unknown}"
echo "  container     : ${GPU_DEV_CONTAINER:-unknown}"
echo "  src dir       : $SRC_DIR"
echo "  branch        : $(git -C "$SRC_DIR" branch --show-current 2>/dev/null)"
echo "  HEAD          : $(git -C "$SRC_DIR" log -1 --format='%h %s' 2>/dev/null)"
echo "  extra op lib  : $EXTRA_OP_LIBRARY_VALUE"
echo "  expert-up op  : $FUSED_EXPERT_UP_OP_VALUE"
echo "  test phase    : $TEST_PHASE_VALUE"
echo "  max tokens    : $MAX_TOKENS_VALUE"
echo "  profile warmup: $PROFILE_WARMUP_VALUE"
echo "  profile iters : $PROFILE_ITERS_VALUE"
echo "  debug output  : $DEBUG_OUTPUT_DIR_VALUE"
echo "  pytest args   : ${PYTEST_ARGS[*]}"
echo "  log           : $LOG"
echo "================================================================="

INNER_CMD="
    set -euo pipefail
    cd ${SRC_DIR_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_EXTRA_OP_LIBRARY=${EXTRA_OP_LIBRARY_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_FUSED_EXPERT_UP_OP=${FUSED_EXPERT_UP_OP_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PHASE=${TEST_PHASE_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_MAX_FUSED_KERNEL_NUM_TOKENS=${MAX_TOKENS_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PROFILE_WARMUP_ITERS=${PROFILE_WARMUP_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PROFILE_ITERS=${PROFILE_ITERS_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_DEBUG_OUTPUT_DIR=${DEBUG_OUTPUT_DIR_Q}
    pytest ${PYTEST_ARGS_Q}
"

set +e
"$EXEC" -- bash -c "$INNER_CMD" 2>&1 | tee "$LOG"
TEST_STATUS="${PIPESTATUS[0]}"
set -e

if [ "$TEST_STATUS" -ne 0 ]; then
    echo "" >&2
    echo "=================================================================" >&2
    echo "  dsv3_fused_expert_up_dev pytest failed with exit code $TEST_STATUS" >&2
    echo "  Test log: $LOG" >&2
    echo "" >&2
    echo "  Tail of log:" >&2
    tail -120 "$LOG" >&2 || true
    echo "=================================================================" >&2
    exit "$TEST_STATUS"
fi

case "$(printf '%s' "$TEST_PHASE_VALUE" | tr '[:upper:]' '[:lower:]')" in
    profile|benchmark|bench|timing|all)
        echo ""
        echo "================================================================="
        echo "  Latest fused MoE profile summary"
        echo "  Source: $PROFILE_SUMMARY"
        if [ -f "$PROFILE_SUMMARY" ]; then
            tail -n 1 "$PROFILE_SUMMARY"
        else
            echo "  Profile summary not found."
        fi
        echo "================================================================="
        ;;
esac

echo ""
echo "================================================================="
echo "  dsv3_fused_expert_up_dev pytest complete"
echo "  Test log: $LOG"
echo "================================================================="
