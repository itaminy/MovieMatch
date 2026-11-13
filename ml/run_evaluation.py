"""Reproducible offline experiment: python -m ml.run_evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from importlib.metadata import version

import pandas as pd

from ml.data_loader import DEFAULT_DATA_DIR, PROJECT_ROOT, load_data
from ml.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "evaluation")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    started = perf_counter()
    try:
        data = load_data(args.data_dir)
        metrics, summary = evaluate(data["movies"], data["ratings"], include_hybrid=True,
                                    progress=lambda msg: print(msg, flush=True))
        summary["data_sha256"] = {name: hashlib.sha256((args.data_dir / f"{name}.csv").read_bytes()).hexdigest()
                                  for name in ("movies", "ratings")}
        summary["versions"] = {"python": platform.python_version(), **{
            name: version(name) for name in ("pandas", "numpy", "scipy", "scikit-learn")}}
        summary["elapsed_seconds"] = perf_counter() - started
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.output_dir / "metrics.csv", index=False)
        pd_validation = pd.DataFrame(summary["validation"]["metrics"])
        pd_validation.to_csv(args.output_dir / "validation_metrics.csv", index=False)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Selected alpha (validation only): {summary['validation']['selected_alpha']}")
    print(pd_validation.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Elapsed: {summary['elapsed_seconds']:.2f} seconds; reports: {args.output_dir.resolve()}")
    short = metrics[metrics["users_below_k"] > 0]
    if not short.empty:
        print("WARNING: some methods have fewer than K candidates; no fallback or padding was used.")
        print(short[["method", "k", "users_below_k"]].to_string(index=False))


if __name__ == "__main__":
    main()
