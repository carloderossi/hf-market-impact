import argparse
import os
import json
import numpy as np
import pandas as pd

from data_generation import simulate_market
from causal_doubleml import run_doubleml
from causal_forest import run_causal_forest


def main():
    parser = argparse.ArgumentParser(description="Run DoubleML + Causal Forest on synthetic HF data")
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    print(f"Generating synthetic market data with {args.n_steps} steps...")
    df = simulate_market(n_steps=args.n_steps)
    df.to_csv(os.path.join(args.output_path, "synthetic_data.csv"), index=False)

    # -------------------------
    # DoubleML
    # -------------------------
    print("Running DoubleML...")
    dml = run_doubleml(df)

    dml_results = {
        "coef": float(dml.coef[0]),
        "se": float(dml.se[0]),
        "t_stat": float(dml.t_stat[0]),
        "p_value": float(dml.pval[0])
    }

    with open(os.path.join(args.output_path, "doubleml_results.json"), "w") as f:
        json.dump(dml_results, f, indent=4)

    with open(os.path.join(args.output_path, "doubleml_summary.txt"), "w") as f:
        f.write(str(dml.summary))

    # -------------------------
    # Causal Forest
    # -------------------------
    print("Running Causal Forest...")
    cf_model, cate = run_causal_forest(df)

    np.save(os.path.join(args.output_path, "causal_forest_cate.npy"), cate)

    cf_results = {
        "cate_mean": float(np.mean(cate)),
        "cate_std": float(np.std(cate)),
        "cate_min": float(np.min(cate)),
        "cate_max": float(np.max(cate))
    }

    with open(os.path.join(args.output_path, "causal_forest_results.json"), "w") as f:
        json.dump(cf_results, f, indent=4)

    print("All models completed successfully.")


if __name__ == "__main__":
    main()
