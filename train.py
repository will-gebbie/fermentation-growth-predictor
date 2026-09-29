#!/usr/bin/env python
"""Benchmark OD soft-sensor models with leave-one-batch-out CV and optionally save one."""

import argparse
import os

import joblib
import pandas as pd

from bioreactor_od.features import (
    DEFAULT_REACTOR_VOLUME,
    build_feature_table,
    get_feature_columns,
)
from bioreactor_od.models import (
    MODEL_TYPES,
    cross_validate,
    fit_model,
    labeled_rows,
    summarize_cv,
)
from bioreactor_od.plots import (
    plot_features_vs_od,
    plot_od_predictions,
    plot_shap_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and benchmark models predicting methanotroph OD from bioreactor off-gas data"
    )
    parser.add_argument(
        "-w",
        "--workbooks",
        nargs="+",
        required=True,
        help="Run workbooks, e.g. data/formatted/*_DATA.xlsx",
    )
    parser.add_argument(
        "--reactor-volume",
        type=float,
        default=DEFAULT_REACTOR_VOLUME,
        help="Broth volume in L, used to express gas rates in mmol/L/h",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        choices=MODEL_TYPES,
        default=MODEL_TYPES,
        help="Models to benchmark with cross-validation",
    )
    parser.add_argument(
        "--save-model",
        choices=MODEL_TYPES,
        help="Train this model on all labeled data and save it as a bundle",
    )
    parser.add_argument(
        "--bundle-path",
        default="models/model_bundle.joblib",
        help="Where to save the trained model bundle",
    )
    parser.add_argument(
        "-o", "--output-dir", default="outputs", help="Directory for plots and results"
    )
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Write the full engineered feature table to CSV",
    )
    parser.add_argument(
        "--plot-features",
        action="store_true",
        help="Plot engineered features vs manual OD for every batch",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_runs_df = build_feature_table(args.workbooks, args.reactor_volume)
    print(
        f"Built {len(get_feature_columns(all_runs_df))} features for "
        f"{all_runs_df['run_id'].nunique()} runs "
        f"({all_runs_df['OD'].notna().sum()} labeled OD samples)"
    )

    if args.save_features:
        all_runs_df.to_csv(f"{args.output_dir}/all_runs_data_w_feats.csv", index=False)

    if args.plot_features:
        for run_id, run_df in all_runs_df.groupby("run_id"):
            plot_features_vs_od(
                run_df, run_id, f"{args.output_dir}/feature_plots/{run_id}_plots"
            )

    summaries = []
    for model_type in args.models:
        print(f"Cross-validating {model_type}...")
        results_df, preds_df = cross_validate(all_runs_df, model_type)
        results_df.to_csv(f"{args.output_dir}/cv_folds_{model_type}.csv", index=False)
        plot_od_predictions(preds_df, model_type, f"{args.output_dir}/cv_plots")
        summaries.append(summarize_cv(model_type, results_df, preds_df))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(f"{args.output_dir}/cv_summary.csv", index=False)
    print("\nLeave-one-batch-out CV summary:")
    print(summary_df.to_string(index=False, float_format="%.3f"))

    if args.save_model:
        od_df = labeled_rows(all_runs_df)
        feature_cols = get_feature_columns(od_df)
        bundle = fit_model(od_df[feature_cols], od_df["OD"], args.save_model)

        bundle_dir = os.path.dirname(args.bundle_path)
        if bundle_dir:
            os.makedirs(bundle_dir, exist_ok=True)
        joblib.dump(bundle, args.bundle_path)
        print(
            f"\nSaved {args.save_model} model to {args.bundle_path} "
            f"({len(od_df)} training samples)"
        )

        if args.save_model == "xgboost":
            plot_shap_summary(bundle, od_df, f"{args.output_dir}/shap_plots")


if __name__ == "__main__":
    main()
