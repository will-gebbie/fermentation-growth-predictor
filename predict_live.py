#!/usr/bin/env python
"""Watch a run workbook during a live reactor run and predict the current OD.

Each time the workbook is updated with new sensor data, the feature pipeline is
re-run and the saved model predicts OD at the most recent timestamp. Predictions are
printed and appended to a CSV log.
"""

import argparse
import os
import time
from datetime import datetime

import joblib
import pandas as pd

from bioreactor_od.features import DEFAULT_REACTOR_VOLUME, build_run_features
from bioreactor_od.models import predict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live OD prediction for a running bioreactor from off-gas data"
    )
    parser.add_argument(
        "-w",
        "--workbook",
        required=True,
        help="Workbook for the current run, updated as new sensor data arrives",
    )
    parser.add_argument(
        "--bundle-path",
        default="models/model_bundle.joblib",
        help="Model bundle saved by train.py --save-model",
    )
    parser.add_argument(
        "--reactor-volume",
        type=float,
        default=DEFAULT_REACTOR_VOLUME,
        help="Broth volume in L (must match the value used in training)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=600,
        help="Seconds between checks for new data (gas analyzer logs every 10 min)",
    )
    parser.add_argument(
        "--log-path",
        default="outputs/live_predictions.csv",
        help="CSV that each prediction is appended to",
    )
    parser.add_argument(
        "--once", action="store_true", help="Predict once and exit instead of watching"
    )
    return parser.parse_args()


def predict_latest(workbook_path, bundle, reactor_volume):
    """Predicted OD at the most recent sensor timestamp in the workbook."""
    features = build_run_features(workbook_path, reactor_volume, clip_to_manual=False)
    latest = features.tail(1)
    predicted_od = predict(bundle, latest)[0]
    return {
        "predicted_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": latest["run_id"].iloc[0],
        "batch": latest["batch"].iloc[0],
        "timestamp": latest["timestamp"].iloc[0],
        "OD_pred": predicted_od,
    }


def append_to_log(row, log_path):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    write_header = not os.path.exists(log_path)
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=write_header, index=False)


def main():
    args = parse_args()
    bundle = joblib.load(args.bundle_path)
    print(f"Loaded {bundle['model_type']} model from {args.bundle_path}")

    last_modified = None
    while True:
        modified = os.path.getmtime(args.workbook)
        if modified != last_modified:
            last_modified = modified
            row = predict_latest(args.workbook, bundle, args.reactor_volume)
            append_to_log(row, args.log_path)
            print(
                f"[{row['predicted_at']}] {row['run_id']} batch {int(row['batch'])} "
                f"@ {row['timestamp']}: predicted OD = {row['OD_pred']:.2f}"
            )

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
