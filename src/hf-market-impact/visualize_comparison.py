import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def load_results(path):
    with open(os.path.join(path, "doubleml_results.json"), "r") as f:
        dml = json.load(f)

    with open(os.path.join(path, "causal_forest_results.json"), "r") as f:
        cf = json.load(f)

    cate = np.load(os.path.join(path, "causal_forest_cate.npy"))
    df = pd.read_csv(os.path.join(path, "synthetic_data.csv"))

    return dml, cf, cate, df


def plot_ate_comparison(dml, cf):
    plt.figure(figsize=(8, 4))
    ate_dml = dml["coef"]
    ate_cf = cf["cate_mean"]

    plt.bar(["DoubleML ATE", "Causal Forest ATE"], [ate_dml, ate_cf],
            color=["#1f77b4", "#ff7f0e"])

    plt.title("Average Treatment Effect: DoubleML vs Causal Forest")
    plt.ylabel("Estimated Effect")
    plt.tight_layout()
    plt.show()


def plot_cate_distribution(cate):
    plt.figure(figsize=(10, 4))
    sns.histplot(cate, bins=50, kde=True, color="#ff7f0e")
    plt.title("Distribution of Causal Forest CATE (Heterogeneous Effects)")
    plt.xlabel("Estimated Treatment Effect")
    plt.tight_layout()
    plt.show()


def plot_cate_vs_features(cate, df):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=df["spread"], y=cate, alpha=0.5)
    plt.title("CATE vs Spread")
    plt.xlabel("Spread")
    plt.ylabel("CATE")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=df["volume"], y=cate, alpha=0.5)
    plt.title("CATE vs Volume")
    plt.xlabel("Volume")
    plt.ylabel("CATE")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize DoubleML vs Causal Forest results")
    parser.add_argument("--results_path", type=str, required=True)
    args = parser.parse_args()

    dml, cf, cate, df = load_results(args.results_path)

    print("Plotting ATE comparison...")
    plot_ate_comparison(dml, cf)

    print("Plotting CATE distribution...")
    plot_cate_distribution(cate)

    print("Plotting CATE vs market features...")
    plot_cate_vs_features(cate, df)


if __name__ == "__main__":
    main()
