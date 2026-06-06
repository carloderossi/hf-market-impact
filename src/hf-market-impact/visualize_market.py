import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_generation import simulate_market


def plot_price(df):
    plt.figure(figsize=(12, 4))
    plt.plot(df["price"], linewidth=1)
    plt.title("Synthetic High-Frequency Price Series")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()


def plot_returns(df):
    plt.figure(figsize=(12, 4))
    sns.histplot(df["return"], bins=100, kde=True)
    plt.title("Distribution of Returns (Fat Tails Expected)")
    plt.xlabel("Return")
    plt.tight_layout()
    plt.show()


def plot_spread(df):
    plt.figure(figsize=(12, 4))
    plt.plot(df["spread"], linewidth=0.8)
    plt.title("Bid-Ask Spread (with occasional widening)")
    plt.xlabel("Time")
    plt.ylabel("Spread")
    plt.tight_layout()
    plt.show()


def plot_volume(df):
    plt.figure(figsize=(12, 4))
    plt.plot(df["volume"], linewidth=0.8)
    plt.title("Trading Volume (clusters around events)")
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.show()


def plot_operations(df):
    plt.figure(figsize=(12, 4))
    plt.plot(df["operation_size"], linewidth=0.8)
    plt.title("Policy Operations (Treatment Variable)")
    plt.xlabel("Time")
    plt.ylabel("Operation Size")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize synthetic HF market data")
    parser.add_argument("--n_steps", type=int, default=20000)
    args = parser.parse_args()

    df = simulate_market(n_steps=args.n_steps)

    print("Plotting synthetic market data...")
    plot_price(df)
    plot_returns(df)
    plot_spread(df)
    plot_volume(df)
    plot_operations(df)


if __name__ == "__main__":
    main()
