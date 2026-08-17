#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""
DGX Spark / Muse-Glimmer-30B-NVFP4-W4A4 Full Benchmark (spark-arena style)
Runs the long-form llama-benchy sweep used for spark-arena-style measurements.

Usage:
  uv run benchmark/benchmark_speed_arena.py
  uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_full.csv
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


COLORS = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
    "dim":    "\033[2m",
}


def c(text, color):
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def header(title):
    line = "─" * 60
    print(f"\n{c(line, 'cyan')}")
    print(f"{c('  ' + title, 'bold')}")
    print(f"{c(line, 'cyan')}")


def result_line(label, value, color="green"):
    print(f"  {c(label.ljust(30), 'dim')} {c(str(value), color)}")


def build_command(args):
    depths = args.depth
    concurrencies = args.concurrency

    if args.verify_128k:
        depths = ["128768"]
        concurrencies = ["1"]

    cmd = [
        "uv",
        "tool",
        "run",
        "--from",
        "llama-benchy",
        "llama-benchy",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--tokenizer",
        args.tokenizer,
        "--depth",
        *depths,
        "--pp",
        str(args.pp),
        "--tg",
        str(args.tg),
        "--enable-prefix-caching",
        "--concurrency",
        *concurrencies,
        "--save-result",
        args.save_result,
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run the full overnight llama-benchy sweep against the local Cogni-Brain vLLM endpoint.",
        epilog="Example: uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_full.csv",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument(
        "--model",
        default="Inferact/Muse-Glimmer-30B-NVFP4-W4A4",
    )
    parser.add_argument("--served-model-name", default="Cogni-Brain")
    parser.add_argument(
        "--tokenizer",
        default="Inferact/Muse-Glimmer-30B-NVFP4-W4A4",
    )
    parser.add_argument("--pp", type=int, default=2048)
    parser.add_argument("--tg", type=int, default=128)
    parser.add_argument(
        "--depth",
        nargs="+",
        default=["0", "4096", "8192", "16384", "32768", "65535", "128768"],
        help="Context depths to test. 128768 + 2048(pp) + 128(tg) = 130944 (tests full 131K context window)",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        default=["1", "2", "5", "10"],
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--verify-128k",
        action="store_true",
        help="Run a 60-second quick smoke test of ONLY the maximum 128K context point (depth=128768, concurrency=1)",
    )
    parser.add_argument("--save-result", default="results_full.csv")
    args = parser.parse_args()

    header("FULL BENCHMARK — Muse-Glimmer-30B-NVFP4-W4A4 (Cogni-Brain)")
    result_line("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result_line("Base URL", args.base_url)
    result_line("Model", args.model)
    result_line("Served model name", args.served_model_name)
    result_line("Tokenizer", args.tokenizer)
    result_line("Depths", " ".join(args.depth) if not args.verify_128k else "128768 (Quick 128K Verify)")
    result_line("Output CSV", args.save_result)
    print()
    print(f"  {c('This sweep tests depths up to 128K context. Prefer to run it overnight.', 'yellow')}")

    if shutil.which("uv") is None:
        print(f"\n{c('✗ uv is not installed or not on PATH', 'red')}")
        sys.exit(1)

    output_path = Path(args.save_result)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_command(args)

    header("COMMAND")
    print("  " + " \\\n    ".join(command))

    header("RUNNING")
    print(f"  {c('Streaming llama-benchy output below...', 'dim')}")

    try:
        completed = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        print(f"\n{c('Benchmark interrupted by user', 'yellow')}")
        sys.exit(130)
    except FileNotFoundError:
        print(f"\n{c('✗ Failed to start uv', 'red')}")
        sys.exit(1)

    if completed.returncode == 0:
        header("DONE")
        result_line("Status", "Success")
        result_line("Saved results", args.save_result)
    else:
        header("FAILED")
        result_line("Exit code", completed.returncode, color="red")
        print(f"  {c('llama-benchy did not complete successfully.', 'red')}")
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
