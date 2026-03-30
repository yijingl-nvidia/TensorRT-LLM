#!/usr/bin/env python3
"""Organize disagg-mode CUDA graph padding experiment results into a tab-separated table.

Suitable for pasting into Google Sheets.

Usage:
    python organize_disagg_cuda_graph_results.py --result-folder disagg_sweep_cuda_graph_KIMI2NVFP4
    python organize_disagg_cuda_graph_results.py --data-dir /full/path/to/logs
"""

import argparse
import json
import os
import re
import sys

CONFIGS = ["no_padding", "default_padding", "padding_slide_64"]
CONCURRENCY_DIR_PATTERN = re.compile(r"^concurrency_(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Organize disagg-mode CUDA graph benchmark results for Google Sheets"
    )
    parser.add_argument(
        "--result-folder",
        default=None,
        help="Result folder name inside $STORAGE_DIR (e.g. disagg_sweep_cuda_graph_KIMI2NVFP4)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Full path to the logs directory containing per-config subfolders (overrides --result-folder)",
    )
    args = parser.parse_args()
    if args.data_dir is None:
        if args.result_folder is None:
            parser.error("Either --result-folder or --data-dir must be specified")
        storage_dir = os.environ.get("STORAGE_DIR", "")
        args.data_dir = os.path.join(storage_dir, args.result_folder, "logs")
    return args


def _read_metrics(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return {
        "throughput": data["output_throughput"],
        "mean_ttft_ms": data["mean_ttft_ms"],
        "p99_ttft_ms": data["p99_ttft_ms"],
        "mean_tpot_ms": data["mean_tpot_ms"],
        "p99_tpot_ms": data["p99_tpot_ms"],
        "mean_e2el_ms": data["mean_e2el_ms"],
        "p99_e2el_ms": data["p99_e2el_ms"],
    }


def load_results(data_dir):
    """Return {config: {concurrency: metrics}}."""
    results = {}
    for config in CONFIGS:
        config_dir = os.path.join(data_dir, config)
        if not os.path.isdir(config_dir):
            print(f"# Warning: config directory not found: {config_dir}", file=sys.stderr)
            continue
        config_results = {}
        for entry in os.listdir(config_dir):
            match = CONCURRENCY_DIR_PATTERN.match(entry)
            if not match:
                continue
            concurrency = int(match.group(1))
            result_file = os.path.join(config_dir, entry, "result.json")
            if not os.path.isfile(result_file):
                print(
                    f"# Warning: missing result.json for {config}/concurrency_{concurrency}",
                    file=sys.stderr,
                )
                continue
            try:
                config_results[concurrency] = _read_metrics(result_file)
            except (KeyError, json.JSONDecodeError) as e:
                print(f"# Warning: failed to parse {result_file}: {e}", file=sys.stderr)
        results[config] = config_results
    return results


def build_table(results):
    all_concurrencies = sorted({c for config_data in results.values() for c in config_data})

    header = ["concurrency"]
    for config in CONFIGS:
        header += [
            f"{config}_TPS",
            f"{config}_mean_ttft_ms",
            f"{config}_p99_ttft_ms",
            f"{config}_mean_tpot_ms",
            f"{config}_p99_tpot_ms",
            f"{config}_mean_e2el_ms",
            f"{config}_p99_e2el_ms",
        ]
        if config != "no_padding":
            header.append(f"{config}_vs_no_padding_%")
    header.append("slide_64_vs_default_%")

    rows = [header]
    for conc in all_concurrencies:
        row = [str(conc)]
        no_padding_entry = results.get("no_padding", {}).get(conc)
        no_padding_tps = no_padding_entry["throughput"] if no_padding_entry else None

        for config in CONFIGS:
            entry = results.get(config, {}).get(conc)
            if entry:
                row += [
                    f"{entry['throughput']:.2f}",
                    f"{entry['mean_ttft_ms']:.2f}",
                    f"{entry['p99_ttft_ms']:.2f}",
                    f"{entry['mean_tpot_ms']:.2f}",
                    f"{entry['p99_tpot_ms']:.2f}",
                    f"{entry['mean_e2el_ms']:.2f}",
                    f"{entry['p99_e2el_ms']:.2f}",
                ]
            else:
                row += ["N/A"] * 7

            if config != "no_padding":
                if entry and no_padding_tps and no_padding_tps > 0:
                    gain_pct = (entry["throughput"] - no_padding_tps) / no_padding_tps * 100
                    row.append(f"{gain_pct:.2f}%")
                else:
                    row.append("N/A")

        slide_entry = results.get("padding_slide_64", {}).get(conc)
        default_entry = results.get("default_padding", {}).get(conc)
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

    if not os.path.isdir(args.data_dir):
        print(f"Directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    results = load_results(args.data_dir)
    if not results:
        print(f"No result files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    for config in CONFIGS:
        count = len(results.get(config, {}))
        print(f"# {config}: {count} concurrency points", file=sys.stderr)
    print(file=sys.stderr)

    rows = build_table(results)
    for row in rows:
        print("\t".join(row))


if __name__ == "__main__":
    main()
