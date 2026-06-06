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
#   scripts/test_dsv3_fused_expert_up_dev_in_session.sh --nsys \
#     -q tests/unittest/_torch/models/test_modeling_deepseekv3_fused_moe.py::test_deepseekv3_fused_moe_profile_phase[0] -s -rs
#
#   scripts/test_dsv3_fused_expert_up_dev_in_session.sh --nsys-query
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
    local task_name="dsv3_fused_expert_up_dev pytest"

    if [ "${RUN_NSYS_QUERY:-0}" -eq 1 ] && [ "${RUN_NSYS:-0}" -eq 0 ]; then
        task_name="dsv3_fused_expert_up_dev nsys query"
    fi

    end_epoch="$(date +%s)"
    elapsed=$((end_epoch - RUN_START_EPOCH))

    echo ""
    echo "================================================================="
    if [ "$status" -eq 0 ]; then
        echo "  ${task_name} OK"
    else
        echo "  ${task_name} FAILED (exit $status)"
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

RUN_NSYS=0
RUN_NSYS_QUERY=0
NSYS_REPORT_VALUE="${TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_NSYS_REPORT:-}"
PYTEST_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --nsys)
            RUN_NSYS=1
            shift
            ;;
        --nsys-query)
            RUN_NSYS_QUERY=1
            shift
            ;;
        --nsys-report)
            [ "$#" -ge 2 ] || {
                echo "ERROR: --nsys-report requires a report path" >&2
                exit 2
            }
            NSYS_REPORT_VALUE="$2"
            shift 2
            ;;
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

BUILD_DIR="${DSV3_FUSED_EXPERT_UP_DEV_BUILD_DIR:-${SRC_DIR}/cpp/build/dsv3_fused_expert_up_dev}"
OUT_SO_DEFAULT="${DSV3_FUSED_EXPERT_UP_DEV_OUTPUT_SO:-${BUILD_DIR}/libdsv3_fused_expert_up_dev.so}"
EXTRA_OP_LIBRARY_VALUE="${TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_EXTRA_OP_LIBRARY:-$OUT_SO_DEFAULT}"
FUSED_EXPERT_UP_OP_VALUE="${TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_FUSED_EXPERT_UP_OP:-dsv3_fused_expert_up_dev}"
TEST_PHASE_DEFAULT="both"
if [ "$RUN_NSYS" -eq 1 ]; then
    TEST_PHASE_DEFAULT="profile"
fi
TEST_PHASE_VALUE="${TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PHASE:-$TEST_PHASE_DEFAULT}"
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
NSYS_OUTPUT_DIR="${TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_NSYS_OUTPUT_DIR:-${OUT_ROOT}/trtllm/nsys_fused_up_dev_$(date +%Y%m%d_%H%M%S)${SESSION_NAME:+_session${SESSION_NAME}}}"
NSYS_OUTPUT_BASENAME="${TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_NSYS_OUTPUT_BASENAME:-fused_up_profile}"

if [ "${#PYTEST_ARGS[@]}" -eq 0 ]; then
    PYTEST_ARGS=(-q tests/unittest/_torch/models/test_modeling_deepseekv3_fused_moe.py -s -rs)
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
NSYS_OUTPUT_DIR_Q="$(shell_quote "$NSYS_OUTPUT_DIR")"
NSYS_OUTPUT_Q="$(shell_quote "${NSYS_OUTPUT_DIR}/${NSYS_OUTPUT_BASENAME}")"

find_latest_nsys_report() {
    local search_root="${OUT_ROOT}/trtllm"
    [ -d "$search_root" ] || return 1
    find "$search_root" -path '*/fused_up_profile*.nsys-rep' -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-
}

run_nsys_query() {
    local report="$1"
    local report_q
    report_q="$(shell_quote "$report")"

    "$EXEC" -- bash -c "
        set -euo pipefail
        REPORT=${report_q}
        echo \"=================================================================\"
        echo \"  Nsight Systems fused expert-up kernel summary\"
        echo \"  Report: \$REPORT\"
        echo \"=================================================================\"
        set +e
        stats_output=\"\$(nsys stats --force-export=true -r cuda_gpu_kern_sum,cuda_kern_exec_sum \"\$REPORT\" 2>&1)\"
        stats_status=\"\$?\"
        set -e
        printf '%s\n' \"\$stats_output\" | grep -E \
            'CUDA GPU Kernel Summary|CUDA Kernel Launch & Exec Time Summary|Time \\(%\\)|PID[[:space:]]+TID|dsv3_fused_expert_up_kernel|Processing|Generating SQLite|NOTICE' || true
        exit \"\$stats_status\"
    "
}

if [ "$RUN_NSYS_QUERY" -eq 1 ] && [ "$RUN_NSYS" -eq 0 ]; then
    if [ -z "$NSYS_REPORT_VALUE" ]; then
        NSYS_REPORT_VALUE="$(find_latest_nsys_report || true)"
    fi
    [ -n "$NSYS_REPORT_VALUE" ] && [ -f "$NSYS_REPORT_VALUE" ] || {
        echo "ERROR: no Nsight Systems report found. Run with --nsys first or pass --nsys-report <path>." >&2
        exit 1
    }
    run_nsys_query "$NSYS_REPORT_VALUE"
    exit 0
fi

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
echo "  nsys          : $RUN_NSYS"
if [ "$RUN_NSYS" -eq 1 ]; then
    echo "  nsys output   : ${NSYS_OUTPUT_DIR}/${NSYS_OUTPUT_BASENAME}.nsys-rep"
fi
echo "  pytest args   : ${PYTEST_ARGS[*]}"
echo "  log           : $LOG"
echo "================================================================="

if [ "$RUN_NSYS" -eq 1 ]; then
    INNER_CMD="
    set -euo pipefail
    cd ${SRC_DIR_Q}
    mkdir -p ${NSYS_OUTPUT_DIR_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_EXTRA_OP_LIBRARY=${EXTRA_OP_LIBRARY_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_FUSED_EXPERT_UP_OP=${FUSED_EXPERT_UP_OP_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PHASE=${TEST_PHASE_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_MAX_FUSED_KERNEL_NUM_TOKENS=${MAX_TOKENS_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PROFILE_WARMUP_ITERS=${PROFILE_WARMUP_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PROFILE_ITERS=${PROFILE_ITERS_Q}
    export TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_DEBUG_OUTPUT_DIR=${DEBUG_OUTPUT_DIR_Q}
    nsys profile --force-overwrite=true --sample=none --trace=cuda,nvtx --output=${NSYS_OUTPUT_Q} \
        --capture-range=none pytest ${PYTEST_ARGS_Q}
"
else
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
fi

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

if grep -Eq "^[0-9]+ skipped" "$LOG" && ! grep -Eq "([0-9]+ failed|[0-9]+ passed)" "$LOG"; then
    echo "" >&2
    echo "=================================================================" >&2
    echo "  pytest completed but all selected tests were skipped" >&2
    echo "  Test log: $LOG" >&2
    echo "  Re-run output includes skip reasons because the default pytest args now include -rs." >&2
    echo "=================================================================" >&2
    exit 2
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

if [ "$RUN_NSYS" -eq 1 ]; then
    NSYS_REPORT_VALUE="${NSYS_OUTPUT_DIR}/${NSYS_OUTPUT_BASENAME}.nsys-rep"
    echo ""
    echo "================================================================="
    echo "  Nsight Systems report"
    echo "  Report: $NSYS_REPORT_VALUE"
    echo "================================================================="
fi

if [ "$RUN_NSYS_QUERY" -eq 1 ]; then
    run_nsys_query "$NSYS_REPORT_VALUE"
fi

echo ""
echo "================================================================="
echo "  dsv3_fused_expert_up_dev pytest complete"
echo "  Test log: $LOG"
echo "================================================================="
