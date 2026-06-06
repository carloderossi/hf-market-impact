# src/hf_impact/data_generation.py
import numpy as np
import pandas as pd

def simulate_market(n_steps=10_000, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1 / (24 * 60)  # minute-ish

    # base price process
    mu = 0.0
    sigma = 0.0005
    eps = rng.normal(size=n_steps)
    price = np.cumsum(mu * dt + sigma * np.sqrt(dt) * eps) + 1.00

    # spreads and volume
    spread = 0.0001 + 0.0002 * (rng.random(n_steps) < 0.05)  # occasional widening
    volume = rng.lognormal(mean=10, sigma=0.5, size=n_steps)

    # policy operations (treatment)
    ops = np.zeros(n_steps)
    op_times = rng.choice(n_steps, size=30, replace=False)
    ops[op_times] = rng.uniform(0.5, 1.5, size=len(op_times))  # operation size

    # impact: temporary drift + vol bump after operations
    impact_window = 30
    impact_mu = np.zeros(n_steps)
    impact_sigma = np.zeros(n_steps)
    for t in op_times:
        end = min(n_steps, t + impact_window)
        impact_mu[t:end] += 0.0002 * ops[t]
        impact_sigma[t:end] += 0.0003 * ops[t]

    # apply impact to price
    eps2 = rng.normal(size=n_steps)
    price_impact = price + np.cumsum(impact_mu * dt + impact_sigma * np.sqrt(dt) * eps2)

    df = pd.DataFrame({
        "t": np.arange(n_steps),
        "price": price_impact,
        "spread": spread,
        "volume": volume,
        "operation_size": ops,
    })
    df["return"] = df["price"].diff().fillna(0)
    return df
