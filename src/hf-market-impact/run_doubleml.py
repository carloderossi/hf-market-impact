import argparse
import os
import json

import pandas as pd

from data_generation import simulate_market
from causal_doubleml import run_doubleml


def main():
    parser = argparse.ArgumentParser(description="Run DoubleML on synthetic HF market data")
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # 1. Generate synthetic data
    print(f"Generating synthetic market data with {args.n_steps} steps...")
    df = simulate_market(n_steps=args.n_steps)

    # 2. Run DoubleML causal inference
    print("Running DoubleML pipeline...")
    dml = run_doubleml(df)

    os.makedirs(args.output_path, exist_ok=True)
    
    # 3. Save results
    print(f"Saving results to '{args.output_path}'...")
    df.to_csv(os.path.join(args.output_path, "synthetic_data.csv"), index=False)

    with open(os.path.join(args.output_path, "doubleml_summary.txt"), "w") as f:
        f.write(str(dml.summary))

    # Save JSON version of the key results
    results = {
        "coef": float(dml.coef[0]), # dml.coef is a 1‑element numpy array.
        "se": float(dml.se),
        "t_stat": float(dml.t_stat),
        "p_value": float(dml.pval)
    }
    with open(os.path.join(args.output_path, "doubleml_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Job completed successfully.")


if __name__ == "__main__":
    main()
