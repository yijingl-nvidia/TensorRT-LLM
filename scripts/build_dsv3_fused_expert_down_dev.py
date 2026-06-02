# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the standalone dsv3_fused_expert_down development Torch ops."""

import os
import shlex
import shutil
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def main() -> None:
    src_dir = Path(os.environ["SRC_DIR"]).resolve()
    build_dir = Path(os.environ["BUILD_DIR"]).resolve()
    out_so = Path(os.environ["OUT_SO"]).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    out_so.parent.mkdir(parents=True, exist_ok=True)

    sources = [
        src_dir / "cpp/tensorrt_llm/thop/dsv3FusedExpertDownDevOp.cpp",
        src_dir / "cpp/tensorrt_llm/kernels/dsv3FusedKernels/dsv3FusedExpertDown.cu",
    ]
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)

    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-DDSV3_FUSED_EXPERT_DOWN_CUDA_NAME=dsv3_fused_expert_down_cuda_dev",
        "-DDSV3_FUSED_EXPERT_DOWN_AR_RESIDUAL_CUDA_NAME=dsv3_fused_expert_down_ar_residual_cuda_dev",
        "-DDSV3_FUSED_EXPERT_DOWN_AR_RESIDUAL_RMS_NORM_CUDA_NAME="
        "dsv3_fused_expert_down_ar_residual_rms_norm_cuda_dev",
    ]
    extra_cuda_cflags.extend(
        shlex.split(os.environ.get("DSV3_FUSED_EXPERT_DOWN_DEV_EXTRA_CUDA_FLAGS", ""))
    )
    extra_ldflags = ["-lcuda"]
    extra_ldflags.extend(
        shlex.split(os.environ.get("DSV3_FUSED_EXPERT_DOWN_DEV_EXTRA_LDFLAGS", ""))
    )

    print("[build] torch", torch.__version__, flush=True)
    print("[build] TORCH_CUDA_ARCH_LIST", os.environ.get("TORCH_CUDA_ARCH_LIST"), flush=True)
    print("[build] sources", flush=True)
    for source in sources:
        print("  ", source, flush=True)
    print("[build] extra_cuda_cflags", extra_cuda_cflags, flush=True)

    start = time.time()
    load(
        name="dsv3_fused_expert_down_dev",
        sources=[str(source) for source in sources],
        build_directory=str(build_dir),
        extra_include_paths=[
            str(src_dir / "cpp"),
            str(src_dir / "cpp/tensorrt_llm/kernels/dsv3FusedKernels"),
        ],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        is_python_module=False,
        verbose=os.environ.get("DSV3_FUSED_EXPERT_DOWN_DEV_VERBOSE", "1") != "0",
    )

    candidates = sorted(
        build_dir.glob("dsv3_fused_expert_down_dev*.so"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        candidates = sorted(build_dir.glob("*.so"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"No built .so found under {build_dir}")

    built_so = candidates[-1]
    if built_so != out_so:
        shutil.copy2(built_so, out_so)

    required_ops = (
        "dsv3_fused_expert_down_dev",
        "dsv3_fused_expert_down_ar_residual_dev",
        "dsv3_fused_expert_down_ar_residual_rms_norm_dev",
    )
    missing_ops = [name for name in required_ops if not hasattr(torch.ops.trtllm, name)]
    if missing_ops:
        raise RuntimeError(f"standalone ops did not register: {missing_ops}")

    print(f"[build] wrote {out_so}", flush=True)
    print(f"[build] elapsed {time.time() - start:.2f}s", flush=True)
    print("[build] test env:", flush=True)
    print(f"  TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_EXTRA_OP_LIBRARY={out_so}", flush=True)
    print(
        "  TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_FUSED_EXPERT_DOWN_OP=dsv3_fused_expert_down_dev",
        flush=True,
    )
    print(
        "  TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_FUSED_EXPERT_DOWN_AR_RESIDUAL_OP="
        "dsv3_fused_expert_down_ar_residual_dev",
        flush=True,
    )
    print(
        "  TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_FUSED_EXPERT_DOWN_AR_RESIDUAL_RMS_NORM_OP="
        "dsv3_fused_expert_down_ar_residual_rms_norm_dev",
        flush=True,
    )


if __name__ == "__main__":
    main()
