"""Export anonymous aggregates from existing EDA/evaluation, without running ML."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from frontend.analytics_data import DEFAULT_SNAPSHOT, METRICS, Snapshot


def prepare(reports, output):
    reports, output = Path(reports), Path(output)
    files = {"eda": reports / "summary.json", "metrics": reports / "evaluation/metrics.csv",
             "validation": reports / "evaluation/validation_metrics.csv",
             "experiment": reports / "evaluation/summary.json"}
    eda = json.loads(files["eda"].read_text(encoding="utf-8"))
    experiment = json.loads(files["experiment"].read_text(encoding="utf-8"))
    metrics = pd.read_csv(files["metrics"])
    validation = pd.read_csv(files["validation"])
    alpha = validation[(validation["k"] == 10) & (validation["method"] == "hybrid")]
    snapshot = {"schema_version": 1, "source": "MovieLens Latest Small; EDA + offline temporal evaluation",
                "overview": {"counts": {key: eda["counts"][key] for key in ("movies", "users", "ratings", "valid_rating_values")},
                             **{key: eda[key] for key in ("mean_rating", "rating_distribution", "genres")},
                             "top_10_movies": [{key: row[key] for key in ("movieId", "title", "rating_count")}
                                               for row in eda["top_10_movies"]]},
                "metrics": metrics[["method", "k", *METRICS]].to_dict("records"),
                "validation": [{"alpha": row.alpha, "ndcg_at_10": row.ndcg_at_k} for row in alpha.itertuples()],
                "selected_alpha": experiment["validation"]["selected_alpha"],
                "source_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}}
    validated = Snapshot.model_validate(snapshot).model_dump()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return validated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_SNAPSHOT.parents[1] / "reports")
    parser.add_argument("--output", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    try:
        prepare(args.reports_dir, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Cannot prepare dashboard: {exc}\nRun python -m ml.eda and python -m ml.run_evaluation first; see README.\n")
    print(f"Dashboard snapshot: {args.output}")


if __name__ == "__main__":
    main()
