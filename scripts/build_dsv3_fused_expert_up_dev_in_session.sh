#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build only the in-development DeepSeek-V3 fused expert-up Torch op inside
# the active gpu_dev_session.
#
# Output:
#   cpp/build/dsv3_fused_expert_up_dev/libdsv3_fused_expert_up_dev.so
#
# Example:
#   scripts/build_dsv3_fused_expert_up_dev_in_session.sh
#   TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_EXTRA_OP_LIBRARY=cpp/build/dsv3_fused_expert_up_dev/libdsv3_fused_expert_up_dev.so \
#   TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_FUSED_EXPERT_UP_OP=dsv3_fused_expert_up_dev \
#     pytest -q tests/unittest/_torch/models/test_modeling_deepseekv3_fused_moe_allreduce.py -s

set -euo pipefail

BUILD_START_EPOCH="$(date +%s)"

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

report_duration() {
    local status="$?"
    local end_epoch elapsed

    end_epoch="$(date +%s)"
    elapsed=$((end_epoch - BUILD_START_EPOCH))

    echo ""
    echo "================================================================="
    if [ "$status" -eq 0 ]; then
        echo "  dsv3_fused_expert_up_dev build OK"
    else
        echo "  dsv3_fused_expert_up_dev build FAILED (exit $status)"
    fi
    echo "  Build duration: $(format_duration "$elapsed")"
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
OUT_SO="${DSV3_FUSED_EXPERT_UP_DEV_OUTPUT_SO:-${BUILD_DIR}/libdsv3_fused_expert_up_dev.so}"
TORCH_CUDA_ARCH_LIST_VALUE="${TORCH_CUDA_ARCH_LIST:-10.0a}"
MAX_JOBS_VALUE="${MAX_JOBS:-${BUILD_JOBS:-32}}"
VERBOSE_VALUE="${DSV3_FUSED_EXPERT_UP_DEV_VERBOSE:-1}"
EXTRA_CUDA_FLAGS_VALUE="${DSV3_FUSED_EXPERT_UP_DEV_EXTRA_CUDA_FLAGS:-}"
EXTRA_LDFLAGS_VALUE="${DSV3_FUSED_EXPERT_UP_DEV_EXTRA_LDFLAGS:-}"
OUT_ROOT="${STORAGE_DIR:-/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_inference/users/${USER}}/nvbug6108841_logs"
RUN_DIR="${OUT_ROOT}/trtllm/fused_up_dev_build_$(date +%Y%m%d_%H%M%S)${SESSION_NAME:+_session${SESSION_NAME}}"
mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/build_dsv3_fused_expert_up_dev.log"

echo "================================================================="
echo "  dsv3_fused_expert_up_dev standalone build"
echo "================================================================="
echo "  session    : job ${GPU_DEV_JOB:-unknown} on ${GPU_DEV_NODE:-unknown}"
echo "  container  : ${GPU_DEV_CONTAINER:-unknown}"
echo "  src dir    : $SRC_DIR"
echo "  branch     : $(git -C "$SRC_DIR" branch --show-current 2>/dev/null)"
echo "  HEAD       : $(git -C "$SRC_DIR" log -1 --format='%h %s' 2>/dev/null)"
echo "  build dir  : $BUILD_DIR"
echo "  output so  : $OUT_SO"
echo "  cuda arch  : $TORCH_CUDA_ARCH_LIST_VALUE"
echo "  max jobs   : $MAX_JOBS_VALUE"
echo "  cuda flags : $EXTRA_CUDA_FLAGS_VALUE"
echo "  ld flags   : $EXTRA_LDFLAGS_VALUE"
echo "  log        : $LOG"
echo "================================================================="

INNER_CMD="
    set -euo pipefail
    export SRC_DIR='$SRC_DIR'
    export BUILD_DIR='$BUILD_DIR'
    export OUT_SO='$OUT_SO'
    export TORCH_CUDA_ARCH_LIST='$TORCH_CUDA_ARCH_LIST_VALUE'
    export MAX_JOBS='$MAX_JOBS_VALUE'
    export DSV3_FUSED_EXPERT_UP_DEV_VERBOSE='$VERBOSE_VALUE'
    export DSV3_FUSED_EXPERT_UP_DEV_EXTRA_CUDA_FLAGS='$EXTRA_CUDA_FLAGS_VALUE'
    export DSV3_FUSED_EXPERT_UP_DEV_EXTRA_LDFLAGS='$EXTRA_LDFLAGS_VALUE'
    cd '$SRC_DIR'
    python3 -u scripts/build_dsv3_fused_expert_up_dev.py
"

set +e
"$EXEC" -- bash -c "$INNER_CMD" 2>&1 | tee "$LOG"
BUILD_STATUS="${PIPESTATUS[0]}"
set -e

if [ "$BUILD_STATUS" -ne 0 ]; then
    echo "" >&2
    echo "=================================================================" >&2
    echo "  Standalone fused expert-up dev build failed with exit code $BUILD_STATUS" >&2
    echo "  Build log: $LOG" >&2
    echo "" >&2
    echo "  Tail of log:" >&2
    tail -120 "$LOG" >&2 || true
    echo "=================================================================" >&2
    exit "$BUILD_STATUS"
fi

echo ""
echo "================================================================="
echo "  Standalone fused expert-up dev build complete"
echo "  Build log: $LOG"
echo "  Output so: $OUT_SO"
echo "================================================================="
