"""Example hybrid recommendations; alpha here is an explicit demonstration setting."""

import argparse
from pathlib import Path
import sys

from ml.data_loader import DEFAULT_DATA_DIR, load_data
from ml.hybrid import HybridRecommender


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        data = load_data(args.data_dir)
        model = HybridRecommender(data["movies"], data["ratings"], alpha=args.alpha)
        for title in ("Toy Story (1995)", "Jumanji (1995)"):
            matches = data["movies"].loc[data["movies"]["title"].eq(title)].sort_values("movieId")
            if matches.empty:
                raise ValueError(f"Movie title not found: {title}")
            print(f"\n{title}, alpha={model.alpha} (demo, not validation tuning):")
            print(model.recommend(int(matches.iloc[0]["movieId"])).to_string(
                index=False, float_format=lambda value: f"{value:.6f}"))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Hybrid demo failed: {exc}\n")


if __name__ == "__main__":
    main()
