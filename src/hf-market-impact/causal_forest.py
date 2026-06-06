# src/hf_impact/causal_forest.py

import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor


def run_causal_forest(df):
    """
    Fit a Causal Forest to estimate heterogeneous treatment effects (CATE)
    of policy operations on returns.

    Parameters
    ----------
    df : pandas.DataFrame
        Synthetic high-frequency market data containing:
        - return
        - operation_size
        - spread
        - volume

    Returns
    -------
    model : CausalForestDML
        Fitted causal forest model.
    cate : np.ndarray
        Estimated conditional average treatment effects for each observation.
    """

    # Outcome (Y)
    y = df["return"].values

    # Treatment (D)
    d = df["operation_size"].values

    # Controls (X)
    x = df[["spread", "volume"]].values

    # Base learners for nuisance functions
    est = CausalForestDML(
        model_t=RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42
        ),
        model_y=RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42
        ),
        n_estimators=500,
        min_samples_leaf=10,
        max_depth=10,
        random_state=42
    )

    # Fit model
    est.fit(
        Y=y,
        T=d,
        X=x
    )

    # Estimate heterogeneous treatment effects
    cate = est.effect(x)

    return est, cate
