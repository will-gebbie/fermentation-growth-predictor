"""Diagnostic plots: features vs OD, CV predictions, and SHAP importance."""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

from bioreactor_od.features import CUMULATIVE_FEATURES, PHYSIO_FEATURES


def plot_features_vs_od(df, run_id, output_dir):
    """One 3x3 grid per batch: each engineered rate over time with manual OD overlaid."""
    os.makedirs(output_dir, exist_ok=True)
    features = list(PHYSIO_FEATURES)
    for feat in CUMULATIVE_FEATURES:
        features.append(f"{feat}_cumsum")

    for batch in df["batch"].unique():
        batch_df = df[df["batch"] == batch]
        od_df = batch_df[batch_df["OD"].notna()]

        fig, axes = plt.subplots(3, 3, figsize=(20, 12))
        fig.suptitle(f"{run_id} batch {int(batch)}")

        for ax, feat in zip(axes.flat, features):
            ax.plot(batch_df["timestamp"], batch_df[feat], color="steelblue")
            ax2 = ax.twinx()
            ax2.scatter(od_df["timestamp"], od_df["OD"], color="red", zorder=5)
            ax.set_title(feat)
            ax2.set_ylabel("OD", color="red")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{run_id}_batch_{int(batch)}_features.png", dpi=100)
        plt.close()


def plot_od_predictions(preds_df, model_name, output_dir):
    """Out-of-fold predicted vs actual OD over time, one plot per run."""
    os.makedirs(output_dir, exist_ok=True)

    for run_id in preds_df["run_id"].unique():
        run_df = preds_df[preds_df["run_id"] == run_id].sort_values("timestamp")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(run_df["timestamp"], run_df["OD"], label="Actual", color="red")
        ax.scatter(
            run_df["timestamp"],
            run_df["OD_pred"],
            label="Predicted",
            color="steelblue",
            marker="x",
        )
        ax.set_title(f"OD Predictions - Run {run_id} ({model_name})")
        ax.set_ylabel("OD")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{run_id}_{model_name}_od_predictions.png", dpi=100)
        plt.close()


def plot_shap_summary(bundle, df, output_dir):
    """SHAP beeswarm summary for a tree-based (XGBoost) model bundle."""
    os.makedirs(output_dir, exist_ok=True)
    X = df[bundle["feature_cols"]]

    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=100, bbox_inches="tight")
    plt.close()
