#!/usr/bin/env python3
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

FULL_BENCHING_BATCH_SIZES = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    16,
    24,
    32,
    33,
    48,
    49,
    64,
    65,
    96,
    97,
    128,
    160,
    192,
    200,
    256,
    320,
    384,
    390,
    396,
    400,
    412,
    416,
    420,
    500,
    512,
    528,
    576,
    608,
    640,
    672,
    704,
    736,
    768,
    900,
    1024,
    1032,
    1040,
    1280,
    1408,
    1536,
    1792,
    1920,
    2000,
]


def _generate_slide_64_batch_sizes(max_batch_size: int) -> list[int]:
    batch_sizes = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
    while True:
        if batch_sizes[-1] + 64 > max_batch_size:
            break
        batch_sizes.append(batch_sizes[-1] + 64)
    return batch_sizes


SLIDE_64_BATCH_SIZES = _generate_slide_64_batch_sizes(2048)


class CudaGraphBenchmark:
    """Orchestrates trtllm-bench throughput benchmarks across CUDA graph padding configs and batch sizes."""

    def __init__(self):
        """Initialize benchmark parameters from environment variables and register signal handlers.

        All parameters are configured via environment variables (defaults in parentheses):

        Model & paths:
            STORAGE_DIR       - Personal storage directory for storing the downloaded LLM model, generated
                                dataset and output files.
            MODEL_NAME        - (required) HuggingFace model name (e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0").
                                Used as the model identifier passed to trtllm-bench and to derive the default
                                model path {STORAGE_DIR}/hf_models/{MODEL_NAME}.
            MODEL_PATH        - (optional) Explicit path to the model directory. If omitted, defaults to
                                {STORAGE_DIR}/hf_models/{MODEL_NAME}.

            OUTPUT_DIR_SUFFIX - Root dir suffix for all output files (e.g. "TinyLlama_h100"). The output dir
                                path will be {STORAGE_DIR}/cuda_graph_testing_logs_{OUTPUT_DIR_SUFFIX}.

        Dataset generation:
            INPUT_LENGTH  (500)  - Fixed input sequence length (ISL) for generated dataset.
            OUTPUT_LENGTH (2000) - Fixed output sequence length (OSL) for generated dataset.
            NUM_DATASET_REQUESTS  (2048) - Number of requests in the generated dataset.

        trtllm-bench engine limits:
            MAX_BATCH_SIZE  (2048) - Maximum runtime batch size passed to --max_batch_size.
            MAX_NUM_TOKENS  (8192) - Maximum runtime token budget passed to --max_num_tokens.

        Parallelism:
            TP_SIZE (1) - Tensor parallelism degree. Passed as --tp when > 1.
            PP_SIZE (1) - Pipeline parallelism degree. Passed as --pp when > 1.

        GPU monitoring:
            GPU_ID           (0) - GPU index for nvidia-smi dmon monitoring.
            MONITOR_INTERVAL (1) - Sampling interval in seconds for nvidia-smi dmon.

        Benchmark mode:
            MODE (sweep) - "sweep" (default) sweeps batch sizes; "variance" repeats a fixed concurrency N times.
            VARIANCE_CONCURRENCY (512) - Concurrency for variance mode.
            NUM_VARIANCE_TRIALS (10) - Number of trials for variance mode.
        """
        # self.model_path = "/tmp/DeepSeek-V3-Lite/fp8"
        # self.model_name = "deepseek_v3_lite_fp8_hf"

        self.model_name = os.environ["MODEL_NAME"]
        self.storage_dir = Path(os.environ["STORAGE_DIR"])
        model_path_override = os.environ.get("MODEL_PATH")
        if model_path_override:
            self.model_path = Path(model_path_override)
        else:
            self.model_path = self.storage_dir / f"hf_models/{self.model_name}"
        output_dir_suffix = os.environ["OUTPUT_DIR_SUFFIX"]
        self.trtllm_code_path = self.storage_dir / "dev/TensorRT-LLM"

        if not self.storage_dir.exists():
            raise ValueError(
                f"Storage directory {self.storage_dir} does not exist! Create it first."
            )
        if not self.model_path.exists():
            raise ValueError(
                f"Model path {self.model_path} does not exist! Download the model first."
            )
        if not self.trtllm_code_path.exists():
            raise ValueError(f"TensorRT-LLM code path {self.trtllm_code_path} does not exist!")

        self.output_dir = self.storage_dir / f"cuda_graph_testing_logs_{output_dir_suffix}"

        print(f"Personal storage directory: {self.storage_dir}")
        print(f"Model path: {self.model_path}")
        print(f"TensorRT-LLM code path: {self.trtllm_code_path}")
        print(f"Output directory: {self.output_dir}")

        self.benching_batch_sizes = FULL_BENCHING_BATCH_SIZES
        # self.benching_batch_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
        # self.benching_batch_sizes = [1, 2, 4, 8, 16, 24, 32, 33, 48, 49]
        # self.benching_batch_sizes = [
        #     128, 160, 192, 200, 256, 320, 384, 390, 396, 400, 412, 416, 420,
        #     500, 512, 528, 576, 608, 640, 672, 704, 736, 768, 900, 1024, 1032,
        #     1040, 1280, 1408, 1536, 1792, 1920, 2000
        # ]
        self.input_length = int(os.environ.get("INPUT_LENGTH", 500))
        self.output_length = int(os.environ.get("OUTPUT_LENGTH", 2000))
        self.num_requests_in_dataset = int(os.environ.get("NUM_DATASET_REQUESTS", 2048))
        self.max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", 2048))
        self.max_num_tokens = int(os.environ.get("MAX_NUM_TOKENS", 8192))
        self.tp_size = int(os.environ.get("TP_SIZE", 1))
        self.pp_size = int(os.environ.get("PP_SIZE", 1))

        self.gpu_id = int(os.environ.get("GPU_ID", 0))
        self.monitor_interval = int(os.environ.get("MONITOR_INTERVAL", 1))
        self.monitor_process: Optional[subprocess.Popen] = None

        # Mode: "sweep" (default) sweeps batch sizes; "variance" repeats a fixed concurrency N times.
        self.mode = os.environ.get("MODE", "sweep")
        if self.mode not in ["sweep", "variance"]:
            raise ValueError(f"Invalid mode: {self.mode}, must be 'sweep' or 'variance'")
        self.variance_concurrency = int(os.environ.get("VARIANCE_CONCURRENCY", 512))
        self.num_variance_trials = int(os.environ.get("NUM_VARIANCE_TRIALS", 10))

        self.setup_logging()

        # SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self.signal_handler)
        # SIGTERM - terminate the process gracefully
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_logging(self):
        """Configure module-level logging with timestamped INFO-level output."""
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

    def signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM by cleaning up subprocesses and exiting."""
        self.logger.info(f"Received signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Terminate the GPU monitoring subprocess if it is still running."""
        self.logger.info("Cleaning up...")
        if self.monitor_process and self.monitor_process.poll() is None:
            self.logger.info(f"Stopping GPU monitoring (PID: {self.monitor_process.pid})")
            self.monitor_process.terminate()
            try:
                self.monitor_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.monitor_process.kill()
                self.monitor_process.wait()
            self.monitor_process = None

    def create_directories(self):
        """Create configs/, logs/, reports/, gpu_logs/, and done_flags/ under output_dir."""
        self.logger.info("Creating output directories...")
        directories = ["configs", "logs", "reports", "gpu_logs", "done_flags"]
        for directory in directories:
            (self.output_dir / directory).mkdir(parents=True, exist_ok=True)

    def _done_flag_path(
        self, config_name: str, batch_size: int, trial: Optional[int] = None
    ) -> Path:
        """Return the path to the done-flag file for a given config, batch size, and optional trial."""
        name = f"done_{config_name}_bs{batch_size}"
        if trial is not None:
            name += f"_trial{trial:02d}"
        return self.output_dir / "done_flags" / name

    def is_run_completed(
        self, config_name: str, batch_size: int, trial: Optional[int] = None
    ) -> bool:
        """Check whether a previous run for this config/batch_size (and optional trial) finished successfully."""
        return self._done_flag_path(config_name, batch_size, trial).exists()

    def create_cuda_graph_configs(self):
        """Write YAML configs for padding-enabled, padding-disabled, and slide-64 CUDA graph modes."""
        self.logger.info("Creating CUDA graph configuration files...")

        # in TRTLLM the batch sizes for CUDA graphs are set in
        # tensorrt_llm/llmapi/llm_args.py:validate_cuda_graph_config()

        # Config 1: Default padding enabled
        config_default = {
            "print_iter_log": False,
            "cuda_graph_config": {
                "enable_padding": True,
                "max_batch_size": self.max_batch_size,
                # Default batch sizes with padding:
                # [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
                # > 128 it will add powers of 2 up to max_batch_size, e.g., (256, 512, 1024, 2048, ... max_batch_size)
            },
            "kv_cache_config": {"dtype": "auto", "free_gpu_memory_fraction": 0.9},
            "enable_chunked_prefill": True,
        }

        with open(self.output_dir / "configs" / "padding_enabled_default.yaml", "w") as f:
            yaml.dump(config_default, f, default_flow_style=False)

        # Config 2: Padding disabled (more comprehensive batch sizes)
        config_disabled = {
            "print_iter_log": False,
            "cuda_graph_config": {
                "enable_padding": False,
                "max_batch_size": self.max_batch_size,
                # Default batch sizes without padding:
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                #  28, 29, 30, 31, 32, 64, 128]
                # > 128 it will add powers of 2 up to max_batch_size, e.g., (256, 512, 1024, 2048, ... max_batch_size)
            },
            "kv_cache_config": {"dtype": "auto", "free_gpu_memory_fraction": 0.9},
            "enable_chunked_prefill": True,
        }

        # Config 3: Padding with slide size 64
        config_slide_64 = {
            "print_iter_log": False,
            "cuda_graph_config": {
                "enable_padding": True,
                "batch_sizes": SLIDE_64_BATCH_SIZES,
            },
            "kv_cache_config": {"dtype": "auto", "free_gpu_memory_fraction": 0.9},
            "enable_chunked_prefill": True,
        }

        with open(self.output_dir / "configs" / "padding_disabled.yaml", "w") as f:
            yaml.dump(config_disabled, f, default_flow_style=False)

        with open(self.output_dir / "configs" / "padding_slide_64.yaml", "w") as f:
            yaml.dump(config_slide_64, f, default_flow_style=False)

        self.logger.info("Created CUDA graph configuration files:")
        config_files = list((self.output_dir / "configs").glob("*.yaml"))
        for config_file in config_files:
            self.logger.info(f"  {config_file}")

    def generate_dataset_if_needed(self) -> Path:
        """Generate a fixed-length tokenized dataset via prepare_dataset.py, or reuse an existing one."""
        dataset_path = (
            self.output_dir
            / f"dataset_{self.input_length}_{self.output_length}_{self.num_requests_in_dataset}.txt"
        )
        if not dataset_path.exists() or dataset_path.stat().st_size == 0:
            self.logger.info(
                f"Generating dataset: ISL={self.input_length}, OSL={self.output_length}, "
                f"requests={self.num_requests_in_dataset}"
            )

            cmd = [
                "python3",
                str(self.trtllm_code_path) + "/benchmarks/cpp/prepare_dataset.py",
                "--stdout",
                "--tokenizer",
                # self.model_name,
                str(self.model_path),
                "--trust-remote-code",
                "token-norm-dist",
                "--num-requests",
                str(self.num_requests_in_dataset),
                "--input-mean",
                str(self.input_length),
                "--input-stdev",
                "0",
                "--output-mean",
                str(self.output_length),
                "--output-stdev",
                "0",
            ]
            print(f"Executing command: {' '.join(cmd)}")

            with open(dataset_path, "w") as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Dataset generation failed: {result.stderr}")

            # Verify the dataset file has content
            if not dataset_path.exists() or dataset_path.stat().st_size == 0:
                raise RuntimeError(
                    f"Dataset generation produced an empty file: {dataset_path}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Stderr: {result.stderr}"
                )

            self.logger.info(f"Dataset generated: {dataset_path}")
        else:
            self.logger.info(f"Using existing dataset: {dataset_path}")

        return dataset_path

    def start_gpu_monitoring(self, config_name: str, batch_size: int, trial: Optional[int] = None):
        """Launch nvidia-smi dmon in the background to log GPU utilization and memory."""
        tag = f"{config_name}_bs{batch_size}"
        if trial is not None:
            tag += f"_trial{trial:02d}"
        gpu_log_path = self.output_dir / "gpu_logs" / f"gpu_monitor_{tag}.log"

        self.logger.info(f"Starting GPU monitoring for {config_name}, batch size {batch_size}")
        self.logger.info(f"GPU log: {gpu_log_path}")

        # nvidia-smi dmon: -s um (utilization,memory), -i (GPU ID), -o T (timestamp), -f (file)
        cmd = [
            "nvidia-smi",
            "dmon",
            "-s",
            "um",
            "-i",
            str(self.gpu_id),
            "-o",
            "T",
            "-f",
            str(gpu_log_path),
        ]

        self.monitor_process = subprocess.Popen(cmd)
        self.logger.info(f"GPU monitoring started with PID: {self.monitor_process.pid}")
        time.sleep(2)

    def stop_gpu_monitoring(self):
        """Terminate the nvidia-smi dmon subprocess, force-killing after 5 s timeout."""
        if self.monitor_process and self.monitor_process.poll() is None:
            self.logger.info(f"Stopping GPU monitoring (PID: {self.monitor_process.pid})")
            self.monitor_process.terminate()
            try:
                self.monitor_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.monitor_process.kill()
                self.monitor_process.wait()
            self.monitor_process = None

    def log_gpu_memory(self, label: str):
        """Log current GPU memory usage (used/free/total) for all GPUs via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self.logger.info(f"[GPU memory] {label}:")
                for line in result.stdout.strip().splitlines():
                    idx, used, free, total = [x.strip() for x in line.split(",")]
                    self.logger.info(
                        f"  GPU {idx}: used={used} MiB, free={free} MiB, total={total} MiB"
                    )
            else:
                self.logger.warning(f"[GPU memory] nvidia-smi failed: {result.stderr.strip()}")
        except Exception as e:
            self.logger.warning(f"[GPU memory] Could not query nvidia-smi: {e}")

    def run_benchmark(
        self,
        config_name: str,
        config_path: Path,
        batch_size: int,
        dataset_path: Path,
        trial: Optional[int] = None,
    ) -> bool:
        """Run a single trtllm-bench throughput test with GPU monitoring.

        Returns True if trtllm-bench finishes successfully, False otherwise.

        Output files created per run:
            {self.output_dir}/logs/benchmark_{config_name}_bs{batch_size}.log
            - Full stdout/stderr from trtllm-bench.
            {self.output_dir}/logs/iteration_{config_name}_bs{batch_size}.log
            - Per-iteration stats (batch size, tokens, timing per scheduler step).
            {self.output_dir}/reports/report_{config_name}_bs{batch_size}.json
            - trtllm-bench output JSON summary: throughput, latency percentiles, and config metadata.
            {self.output_dir}/gpu_logs/gpu_monitor_{config_name}_bs{batch_size}.log
            - nvidia-smi dmon output (GPU utilization & memory usage over time).
        """
        tag = f"{config_name}_bs{batch_size}"
        if trial is not None:
            tag += f"_trial{trial:02d}"
        benchmark_log = self.output_dir / "logs" / f"benchmark_{tag}.log"
        report_json = self.output_dir / "reports" / f"report_{tag}.json"

        self.logger.info(f"Running benchmark: {config_name}, batch size: {batch_size}")
        self.logger.info(f"Config: {config_path}")
        self.logger.info(f"Dataset: {dataset_path}")
        self.logger.info(f"Benchmark log: {benchmark_log}")
        self.logger.info(f"Report: {report_json}")

        self.start_gpu_monitoring(config_name, batch_size, trial)

        # Build benchmark command
        iteration_log = self.output_dir / "logs" / f"iteration_{tag}.log"
        # when batch_size is small (e.g. 1), limit the number of total requests in benchmarking to be at most
        # batch_size * 32.
        runtime_total_requests = min(self.num_requests_in_dataset, batch_size * 32)
        cmd = ["mpirun", "-n", "1", "--oversubscribe", "--allow-run-as-root"]
        cmd += [
            # "trtllm-llmapi-launch",
            "trtllm-bench",
            f"--model={self.model_name}",
            f"--model_path={self.model_path}",
            "throughput",
            f"--dataset={dataset_path}",
            "--backend=pytorch",
            f"--max_batch_size={self.max_batch_size}",
            f"--max_num_tokens={self.max_num_tokens}",
            f"--concurrency={batch_size}",
            f"--num_requests={runtime_total_requests}",
            f"--extra_llm_api_options={config_path}",
            f"--report_json={report_json}",
            f"--iteration_log={iteration_log}",
        ]

        # Add parallelism options if specified
        if self.tp_size > 1:
            cmd.append(f"--tp={self.tp_size}")
        if self.pp_size > 1:
            cmd.append(f"--pp={self.pp_size}")

        self.logger.info(f"Executing: {' '.join(cmd)}")

        try:
            self.log_gpu_memory(f"before trtllm-bench {config_name} bs{batch_size}")
            with open(benchmark_log, "w") as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            self.log_gpu_memory(f"after trtllm-bench {config_name} bs{batch_size}")

            if result.returncode == 0:
                self.logger.info(
                    f"Benchmark completed successfully for {config_name}, batch size {batch_size}"
                )
                self._done_flag_path(config_name, batch_size, trial).touch()
                success = True
            else:
                self.logger.error(
                    f"Benchmark failed for {config_name}, batch size {batch_size}. Check {benchmark_log}"
                )
                success = False

        except Exception as e:
            self.logger.error(f"Exception during benchmark: {e}")
            success = False
        finally:
            self.stop_gpu_monitoring()
            time.sleep(5)

        return success

    def run_variance(self, configs: dict, dataset_path: Path):
        """Run the variance mode: repeat a fixed concurrency NUM_VARIANCE_TRIALS times per config."""
        self.logger.info(
            f"Variance mode: concurrency={self.variance_concurrency}, trials={self.num_variance_trials}"
        )
        total_tests = len(configs) * self.num_variance_trials
        current_test = 0

        for config_name, config_path in configs.items():
            self.logger.info(f"Testing configuration: {config_name}")

            for trial in range(1, self.num_variance_trials + 1):
                current_test += 1

                if self.is_run_completed(config_name, self.variance_concurrency, trial):
                    self.logger.info(
                        f"Progress: {current_test}/{total_tests} - "
                        f"Skipping {config_name} trial {trial:02d} (already completed)"
                    )
                    continue

                self.logger.info(
                    f"Progress: {current_test}/{total_tests} - "
                    f"{config_name} trial {trial:02d}/{self.num_variance_trials}"
                )

                if self.run_benchmark(
                    config_name, config_path, self.variance_concurrency, dataset_path, trial
                ):
                    self.logger.info(f"✓ Completed: {config_name}, trial {trial:02d}")
                else:
                    self.logger.error(f"✗ Failed: {config_name}, trial {trial:02d}")

        self.logger.info("Variance analysis completed!")
        self.logger.info(f"Results available in: {self.output_dir}")

    def run(self):
        """Execute the full benchmark sweep: setup, dataset generation, and all config x batch_size runs.

        Steps:
            1. Create output directory structure under {self.output_dir}/.
            2. Generate CUDA graph YAML configs (padding-enabled, padding-disabled, slide-64).
            3. Generate (or reuse) a tokenized dataset with fixed ISL/OSL.
            4. For each active config x batch_size combination, call run_benchmark() which
               launches trtllm-bench with GPU monitoring.

        Output directory layout after a complete run:
            {self.output_dir}/
            ├── configs/
            │   ├── padding_enabled_default.yaml  - CUDA graph config with default padding buckets.
            │   ├── padding_disabled.yaml          - CUDA graph config with padding disabled.
            │   └── padding_slide_64.yaml          - CUDA graph config with stride-64 batch size buckets.
            ├── dataset_{{isl}}_{{osl}}_{{n}}.txt      - Tokenized dataset (shared across all runs).
            ├── logs/
            │   ├── benchmark_{{cfg}}_bs{{bs}}.log     - Full stdout/stderr from trtllm-bench.
            │   └── iteration_{{cfg}}_bs{{bs}}.log     - Per-iteration scheduler stats (batch size, tokens, timing).
            ├── reports/
            │   └── report_{{cfg}}_bs{{bs}}.json       - JSON summary: throughput, latency percentiles, config metadata.
            ├── gpu_logs/
            │   └── gpu_monitor_{{cfg}}_bs{{bs}}.log   - nvidia-smi dmon output (GPU utilization & memory over time).
            └── done_flags/
                └── done_{{cfg}}_bs{{bs}}               - Empty flag file written on successful completion
                                                          (enables resume).
        """
        self.logger.info("Starting CUDA Graph Padding Analysis")
        self.logger.info(f"Mode: {self.mode}")
        self.logger.info(f"Output directory: {self.output_dir}")

        try:
            self.create_directories()
            self.create_cuda_graph_configs()

            dataset_path = self.generate_dataset_if_needed()
            configs = {
                "default_padding": self.output_dir / "configs" / "padding_enabled_default.yaml",
                "no_padding": self.output_dir / "configs" / "padding_disabled.yaml",
                "padding_slide_64": self.output_dir / "configs" / "padding_slide_64.yaml",
            }

            if self.mode == "variance":
                self.run_variance(configs, dataset_path)
                return

            total_tests = len(configs) * len(self.benching_batch_sizes)
            current_test = 0

            for config_name, config_path in configs.items():
                self.logger.info(f"Testing configuration: {config_name}")

                for batch_size in self.benching_batch_sizes:
                    current_test += 1

                    if self.is_run_completed(config_name, batch_size):
                        self.logger.info(
                            f"Progress: {current_test}/{total_tests} - "
                            f"Skipping {config_name} bs {batch_size} (already completed)"
                        )
                        continue

                    self.logger.info(
                        f"Progress: {current_test}/{total_tests} - Testing {config_name} with batch size {batch_size}"
                    )

                    if self.run_benchmark(config_name, config_path, batch_size, dataset_path):
                        self.logger.info(f"✓ Completed: {config_name}, batch size {batch_size}")
                    else:
                        self.logger.error(f"✗ Failed: {config_name}, batch size {batch_size}")

            self.logger.info("CUDA Graph Padding Analysis completed!")
            self.logger.info(f"Results available in: {self.output_dir}")

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise
        finally:
            self.cleanup()


def main():
    """Entry point: instantiate CudaGraphBenchmark and run the full analysis."""
    parser = argparse.ArgumentParser(
        description="CUDA graph padding benchmark and analysis tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (can also be set via these flags):
  MODEL_NAME          HuggingFace model name, e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (required)
  STORAGE_DIR         Root storage directory (required)
  MODEL_PATH          Explicit model directory path; defaults to $STORAGE_DIR/hf_models/$MODEL_NAME (optional)
  OUTPUT_DIR_SUFFIX   Suffix appended to the output directory name (required)
  NUM_GPUS            Number of GPUs (default: 1)
  GPU_MEMORY_FRAC     GPU memory fraction (default: 0.95)
  MAX_BATCH_SIZE      Maximum batch size (default: 2048)
  MAX_NUM_TOKENS      Maximum number of tokens (default: 8192)
  ISL                 Input sequence length (default: 128)
  OSL                 Output sequence length (default: 128)
  CUDA_GRAPHS         Enable CUDA graphs: "1"/"true" (default: true)
  MODE                Benchmark mode: "sweep" or "variance" (default: sweep)
  VARIANCE_CONCURRENCY  Concurrency for variance mode (default: 512)
  NUM_VARIANCE_TRIALS   Number of trials for variance mode (default: 10)
        """,
    )
    parser.add_argument(
        "--model-name", metavar="NAME", help="HuggingFace model name, required (sets MODEL_NAME)"
    )
    parser.add_argument("--storage-dir", metavar="DIR", help="Storage directory (sets STORAGE_DIR)")
    parser.add_argument(
        "--model-path", metavar="PATH", help="Override model path (sets MODEL_PATH)"
    )
    parser.add_argument(
        "--output-dir-suffix",
        metavar="SUFFIX",
        help="Output directory suffix (sets OUTPUT_DIR_SUFFIX)",
    )
    args = parser.parse_args()

    # Allow CLI args to set env vars so __init__ picks them up.
    # Note: os.environ changes only affect this process and its children —
    # they do NOT persist in the parent shell after the script exits.
    if args.model_name:
        os.environ["MODEL_NAME"] = args.model_name
    if args.storage_dir:
        os.environ["STORAGE_DIR"] = args.storage_dir
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    if args.output_dir_suffix:
        os.environ["OUTPUT_DIR_SUFFIX"] = args.output_dir_suffix

    benchmark = CudaGraphBenchmark()
    benchmark.run()


if __name__ == "__main__":
    main()
