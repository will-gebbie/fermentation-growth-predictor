"""Model fitting, leave-one-batch-out cross-validation and prediction.

A trained model is stored as a "bundle" dict holding everything needed to predict:
the estimator, the feature column order, and any preprocessing (scaler, medians).
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from bioreactor_od.features import get_feature_columns

MODEL_TYPES = ["ridge", "pls", "xgboost"]

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
MAX_PLS_COMPONENTS = 10
XGB_PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200],
    "subsample": [0.8, 1.0],
}


def fit_model(X, y, model_type):
    """Fit one model type and return its bundle.

    XGBoost handles missing values natively and is trained on log(OD) since growth is
    roughly exponential. Ridge and PLS get median imputation and standard scaling.
    """
    feature_cols = list(X.columns)

    if model_type == "xgboost":
        search = GridSearchCV(
            XGBRegressor(random_state=42),
            XGB_PARAM_GRID,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        search.fit(X, np.log(y))
        return {
            "model_type": model_type,
            "model": search.best_estimator_,
            "feature_cols": feature_cols,
            "log_target": True,
        }

    medians = X.median()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(medians).fillna(0))

    if model_type == "ridge":
        model = RidgeCV(alphas=RIDGE_ALPHAS)
    elif model_type == "pls":
        n_components = min(MAX_PLS_COMPONENTS, X_scaled.shape[1], X_scaled.shape[0] - 1)
        model = PLSRegression(n_components=n_components)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_scaled, y)
    return {
        "model_type": model_type,
        "model": model,
        "scaler": scaler,
        "medians": medians,
        "feature_cols": feature_cols,
        "log_target": False,
    }


def predict(bundle, df):
    """Predict OD for every row of df using a model bundle."""
    X = df[bundle["feature_cols"]]
    if "scaler" in bundle:
        X = bundle["scaler"].transform(X.fillna(bundle["medians"]).fillna(0))

    preds = np.ravel(bundle["model"].predict(X))
    if bundle["log_target"]:
        preds = np.exp(preds)
    return preds


def labeled_rows(df):
    """Rows that have a manual OD measurement."""
    return df[df["OD"].notna()].copy()


def cross_validate(df, model_type):
    """Leave-one-batch-out CV across all runs.

    Returns per-fold metrics and the out-of-fold predictions for every labeled row.
    """
    od_df = labeled_rows(df)
    feature_cols = get_feature_columns(od_df)
    folds = od_df[["run_id", "batch"]].drop_duplicates()

    results = []
    all_preds = []
    for fold in folds.itertuples(index=False):
        test_mask = (od_df["run_id"] == fold.run_id) & (od_df["batch"] == fold.batch)
        train_df = od_df[~test_mask]
        test_df = od_df[test_mask]

        bundle = fit_model(train_df[feature_cols], train_df["OD"], model_type)
        y_pred = predict(bundle, test_df)

        results.append(
            {
                "run_id": fold.run_id,
                "batch": fold.batch,
                "n_samples": len(test_df),
                "MAE": mean_absolute_error(test_df["OD"], y_pred),
            }
        )

        pred_df = test_df[["run_id", "batch", "timestamp", "OD"]].copy()
        pred_df["OD_pred"] = y_pred
        all_preds.append(pred_df)

    return pd.DataFrame(results), pd.concat(all_preds, ignore_index=True)


def summarize_cv(model_type, results_df, preds_df):
    """One-row summary of a model's cross-validation performance."""
    return {
        "model": model_type,
        "pooled_MAE": mean_absolute_error(preds_df["OD"], preds_df["OD_pred"]),
        "pooled_R2": r2_score(preds_df["OD"], preds_df["OD_pred"]),
        "mean_fold_MAE": results_df["MAE"].mean(),
        "std_fold_MAE": results_df["MAE"].std(),
        "n_folds": len(results_df),
        "n_samples": len(preds_df),
    }
