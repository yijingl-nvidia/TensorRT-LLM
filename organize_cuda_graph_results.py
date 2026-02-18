#!/usr/bin/env python3
"""Organize CUDA graph padding experiment results into a tab-separated table.

Suitable for pasting into Google Sheets.

Usage:
    python scripts/organize_cuda_graph_results.py [--data-dir DIR]
"""

import argparse
import json
import os
import re
import sys

DEFAULT_DATA_DIR = (
    "/lustre/fsw/coreai_comparch_trtllm/yijingl/cuda_graph_testing_logs_TinyLlama_h100/reports"
)

CONFIGS = ["no_padding", "default_padding", "padding_slide_64"]
FILENAME_PATTERN = re.compile(r"^report_(.+)_bs(\d+)\.json$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Organize CUDA graph benchmark results for Google Sheets"
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing report JSON files"
    )
    return parser.parse_args()


def load_results(data_dir) -> dict[str, dict[int, dict[str, float]]]:
    """Load all JSON report files and return a nested dict."""
    results = {}
    for filename in os.listdir(data_dir):
        match = FILENAME_PATTERN.match(filename)
        if not match:
            continue
        config = match.group(1)
        batch_size = int(match.group(2))

        filepath = os.path.join(data_dir, filename)
        with open(filepath) as f:
            data = json.load(f)

        throughput = data["performance"]["system_output_throughput_tok_s"]
        avg_concurrent = data["request_info"]["avg_num_concurrent_requests"]

        results.setdefault(config, {})[batch_size] = {
            "throughput": throughput,
            "avg_concurrent": avg_concurrent,
        }

    return results


def build_table(results):
    """Build a list of rows (each row is a list of strings) for the TSV output."""
    # Collect all batch sizes across all configs
    all_batch_sizes = set()
    for config_data in results.values():
        all_batch_sizes.update(config_data.keys())
    all_batch_sizes = sorted(all_batch_sizes)

    # Header row
    header = ["batch_size"]
    for config in CONFIGS:
        header.append(f"{config}_avg_conc_req")
        header.append(f"{config}_TPS")
        if config != "no_padding":
            header.append(f"{config}_vs_no_padding_%")
    header.append("slide_64_vs_default_%")

    rows = [header]

    for bs in all_batch_sizes:
        row = [str(bs)]
        no_padding_tp = results.get("no_padding", {}).get(bs, {}).get("throughput")

        for config in CONFIGS:
            entry = results.get(config, {}).get(bs)
            if entry:
                row.append(f"{entry['avg_concurrent']:.2f}")
                row.append(f"{entry['throughput']:.2f}")
            else:
                row.append("N/A")
                row.append("N/A")

            if config != "no_padding":
                if entry and no_padding_tp and no_padding_tp > 0:
                    gain_pct = (entry["throughput"] - no_padding_tp) / no_padding_tp * 100
                    row.append(f"{gain_pct:.2f}%")
                else:
                    row.append("N/A")

        # slide_64 vs default_padding
        slide_entry = results.get("padding_slide_64", {}).get(bs)
        default_entry = results.get("default_padding", {}).get(bs)
        if slide_entry and default_entry and default_entry["throughput"] > 0:
            gain_pct = (
                (slide_entry["throughput"] - default_entry["throughput"])
                / default_entry["throughput"]
                * 100
            )
            row.append(f"{gain_pct:.2f}%")
        else:
            row.append("N/A")

        rows.append(row)

    return rows


def main():
    args = parse_args()
    results = load_results(args.data_dir)

    if not results:
        print(f"No report files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    for config in CONFIGS:
        count = len(results.get(config, {}))
        print(f"# {config}: {count} files", file=sys.stderr)
    print(file=sys.stderr)

    rows = build_table(results)

    # Print tab-separated output (copy-paste into Google Sheets)
    for row in rows:
        print("\t".join(row))


if __name__ == "__main__":
    main()
