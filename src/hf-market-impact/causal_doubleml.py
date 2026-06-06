# src/hf_impact/causal_doubleml.py

import numpy as np
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor


def run_doubleml(df):
    """
    Run a DoubleML partially linear regression (PLR) model
    to estimate the causal effect of policy operations on returns.

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
    dml_plr : DoubleMLPLR
        Fitted DoubleML model with summary, coef, p-values, etc.
    """

    # Outcome (Y)
    y = df["return"].values

    # Treatment (D)
    d = df["operation_size"].values

    # Controls (X)
    x = df[["spread", "volume"]].values

    # Wrap in DoubleML data structure
    data = DoubleMLData.from_arrays(x, y, d)

    # ML learners for nuisance functions
    ml_g = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    )

    ml_m = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    )

    # DoubleML PLR model
    dml_plr = DoubleMLPLR(
        data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=5
    )

    # Fit model
    dml_plr.fit()

    return dml_plr
