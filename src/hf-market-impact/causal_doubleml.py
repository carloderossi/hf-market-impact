# src/hf_impact/causal_doubleml.py

import numpy as np
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor


def run_doubleml(df):
    """
    Run a DoubleML partially linear regression (PLR) model
    to estimate the causal effect of policy operations on returns.
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

    # NEW: ml_l must be provided explicitly
    ml_l = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    )

    # DoubleML PLR model (new API)
    dml_plr = DoubleMLPLR(
        data,
        # ml_g=ml_g, # ml_g is ignored unless you use a different score.
        ml_m=ml_m, # (model for D | X)
        ml_l=ml_l, # (model for D | X in the score)
        n_folds=5
    )

    dml_plr.fit()

    return dml_plr
