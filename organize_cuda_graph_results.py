#!/usr/bin/env python3
"""Organize CUDA graph padding experiment results into a tab-separated table.

Suitable for pasting into Google Sheets.

Normal mode (one result per config+bs):
    python scripts/organize_cuda_graph_results.py --result-folder FOLDER

Variance mode (multiple trials per config+bs, filenames end in _trial{N:02d}.json):
    python scripts/organize_cuda_graph_results.py --result-folder FOLDER
"""

import argparse
import json
import math
import os
import re
import sys

CONFIGS = ["no_padding", "default_padding", "padding_slide_64"]
FILENAME_PATTERN = re.compile(r"^report_(.+)_bs(\d+)\.json$")
VARIANCE_FILENAME_PATTERN = re.compile(r"^report_(.+)_bs(\d+)_trial(\d+)\.json$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Organize CUDA graph benchmark results for Google Sheets"
    )
    parser.add_argument(
        "--result-folder",
        required=True,
        help="Result folder name inside $STORAGE_DIR (e.g. cuda_graph_testing_logs_DSR1NVFP4_gb200)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Full path to directory containing report JSON files (overrides --result-folder)",
    )
    args = parser.parse_args()
    if args.data_dir is None:
        storage_dir = os.environ.get("STORAGE_DIR", "")
        args.data_dir = os.path.join(storage_dir, args.result_folder, "reports")
    return args


def _read_metrics(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return {
        "throughput": data["performance"]["system_output_throughput_tok_s"],
        "avg_concurrent": data["request_info"]["avg_num_concurrent_requests"],
    }


def is_variance_mode(data_dir):
    return any(VARIANCE_FILENAME_PATTERN.match(f) for f in os.listdir(data_dir))


# ---------------------------------------------------------------------------
# Normal mode
# ---------------------------------------------------------------------------


def load_results(data_dir):
    results = {}
    for filename in os.listdir(data_dir):
        match = FILENAME_PATTERN.match(filename)
        if not match:
            continue
        config, batch_size = match.group(1), int(match.group(2))
        results.setdefault(config, {})[batch_size] = _read_metrics(os.path.join(data_dir, filename))
    return results


def build_table(results):
    all_batch_sizes = sorted({bs for config_data in results.values() for bs in config_data})

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


# ---------------------------------------------------------------------------
# Variance mode
# ---------------------------------------------------------------------------


def load_variance_results(data_dir):
    """Return {config: {bs: {trial_num: metrics}}}."""
    results = {}
    for filename in sorted(os.listdir(data_dir)):
        match = VARIANCE_FILENAME_PATTERN.match(filename)
        if not match:
            continue
        config, bs, trial = match.group(1), int(match.group(2)), int(match.group(3))
        results.setdefault(config, {}).setdefault(bs, {})[trial] = _read_metrics(
            os.path.join(data_dir, filename)
        )
    return results


def _stdev(values):
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))


def build_variance_table(results):
    all_batch_sizes = sorted({bs for config_data in results.values() for bs in config_data})

    rows = []
    for bs in all_batch_sizes:
        if rows:
            rows.append([""])  # blank separator between batch size groups

        rows.append([f"=== batch_size={bs} ==="])

        all_trials = sorted(
            {trial for config in CONFIGS for trial in results.get(config, {}).get(bs, {})}
        )

        # Header
        header = ["trial"]
        for config in CONFIGS:
            header.append(f"{config}_avg_conc")
            header.append(f"{config}_TPS")
        rows.append(header)

        # One row per trial
        for trial in all_trials:
            row = [f"trial{trial:02d}"]
            for config in CONFIGS:
                entry = results.get(config, {}).get(bs, {}).get(trial)
                if entry:
                    row.append(f"{entry['avg_concurrent']:.2f}")
                    row.append(f"{entry['throughput']:.2f}")
                else:
                    row.append("N/A")
                    row.append("N/A")
            rows.append(row)

        # Summary rows: mean, stddev, cv%
        mean_row = ["mean"]
        stddev_row = ["stddev"]
        cv_row = ["cv%"]
        for config in CONFIGS:
            trials = results.get(config, {}).get(bs, {})
            if trials:
                concs = [v["avg_concurrent"] for v in trials.values()]
                tps = [v["throughput"] for v in trials.values()]
                mean_conc = sum(concs) / len(concs)
                mean_tps = sum(tps) / len(tps)
                mean_row.extend([f"{mean_conc:.2f}", f"{mean_tps:.2f}"])

                sd_conc = _stdev(concs)
                sd_tps = _stdev(tps)
                stddev_row.append(f"{sd_conc:.2f}" if sd_conc is not None else "N/A")
                stddev_row.append(f"{sd_tps:.2f}" if sd_tps is not None else "N/A")

                cv = sd_tps / mean_tps * 100 if (sd_tps is not None and mean_tps > 0) else None
                cv_row.append("")  # no cv for avg_conc
                cv_row.append(f"{cv:.2f}%" if cv is not None else "N/A")
            else:
                mean_row.extend(["N/A", "N/A"])
                stddev_row.extend(["N/A", "N/A"])
                cv_row.extend(["", "N/A"])

        rows.extend([mean_row, stddev_row, cv_row])

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    if is_variance_mode(args.data_dir):
        results = load_variance_results(args.data_dir)
        if not results:
            print(f"No variance report files found in {args.data_dir}", file=sys.stderr)
            sys.exit(1)
        print("# Variance mode", file=sys.stderr)
        for config in CONFIGS:
            count = sum(len(trials) for trials in results.get(config, {}).values())
            print(f"# {config}: {count} trial files", file=sys.stderr)
        print(file=sys.stderr)
        rows = build_variance_table(results)
    else:
        results = load_results(args.data_dir)
        if not results:
            print(f"No report files found in {args.data_dir}", file=sys.stderr)
            sys.exit(1)
        for config in CONFIGS:
            count = len(results.get(config, {}))
            print(f"# {config}: {count} files", file=sys.stderr)
        print(file=sys.stderr)
        rows = build_table(results)

    for row in rows:
        print("\t".join(row))


if __name__ == "__main__":
    main()
