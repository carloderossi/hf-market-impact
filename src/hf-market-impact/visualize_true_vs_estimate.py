import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Compare true effect vs estimated effects")
    parser.add_argument("--results_path", type=str, required=True)
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(os.path.join(args.results_path, "synthetic_data.csv"))

    with open(os.path.join(args.results_path, "doubleml_results.json")) as f:
        dml = json.load(f)

    with open(os.path.join(args.results_path, "causal_forest_results.json")) as f:
        cf = json.load(f)

    cate = np.load(os.path.join(args.results_path, "causal_forest_cate.npy"))

    # Extract values
    true_ate = df["true_effect"].mean()
    dml_ate = dml["coef"]
    cf_ate = cf["cate_mean"]

    # Bar chart
    plt.figure(figsize=(8, 4))
    plt.bar(["True ATE", "DoubleML ATE", "Causal Forest ATE"],
            [true_ate, dml_ate, cf_ate],
            color=["#2ca02c", "#1f77b4", "#ff7f0e"])
    plt.title("True Effect vs Estimated Effects")
    plt.ylabel("Effect")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

## cls; & "C:/Program Files/Python313/python.exe" c:/Carlo/projects/hf-market-impact/src/hf-market-impact/visualize_true_vs_estimate.py --results_path C:\Carlo\projects\hf-market-impact\outputs\test_run
## cls; & "C:/Program Files/Python313/python.exe" c:/Carlo/projects/hf-market-impact/src/hf-market-impact/visualize_true_vs_estimate.py --results_path C:\Carlo\projects\hf-market-impact\outputs\az-ml-runs