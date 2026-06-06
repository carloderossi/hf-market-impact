import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Heatmap of CATE vs Spread and Volume")
    parser.add_argument("--results_path", type=str, required=True)
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(os.path.join(args.results_path, "synthetic_data.csv"))
    cate = np.load(os.path.join(args.results_path, "causal_forest_cate.npy"))

    df["cate"] = cate

    # Bin spread and volume
    df["spread_bin"] = pd.qcut(df["spread"], 10, duplicates="drop")
    df["volume_bin"] = pd.qcut(df["volume"], 10, duplicates="drop")

    # Pivot table
    heatmap_data = df.pivot_table(
        index="spread_bin",
        columns="volume_bin",
        values="cate",
        aggfunc="mean"
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="coolwarm", center=0)
    plt.title("CATE Heatmap: Treatment Effect vs Spread & Volume")
    plt.xlabel("Volume (binned)")
    plt.ylabel("Spread (binned)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

## cls; & "C:/Program Files/Python313/python.exe" c:/Carlo/projects/hf-market-impact/src/hf-market-impact/visualize_cate_heatmap.py --results_path C:\Carlo\projects\hf-market-impact\outputs\test_run
## cls; & "C:/Program Files/Python313/python.exe" c:/Carlo/projects/hf-market-impact/src/hf-market-impact/visualize_cate_heatmap.py --results_path C:\Carlo\projects\hf-market-impact\outputs\az-ml-runs