"""Time-series feature engineering from off-gas and process signals.

All windows are in rows of the 1-minute grid produced by data.merge_sources, and all
per-batch features are grouped by batch so they reset at each harvest/refill.
"""

import pandas as pd

from bioreactor_od.data import load_workbook, merge_sources

GAS_COLS = ["O2_in", "O2_out", "CH4_in", "CH4_out", "CO2_out"]
SMOOTH_WINDOW = 5

PHYSIO_FEATURES = ["OUR", "CER", "MUR", "RQ", "O2_CH4", "carbon_recovery"]
CUMULATIVE_FEATURES = ["OUR", "CER", "MUR"]
LAG_INTERVALS = [15, 45, 60, 90, 120]
ROLLING_WINDOWS = [15, 30, 60]

NON_FEATURE_COLS = {"OD", "run_id", "batch", "timestamp", "notes"}

# Molar volume of an ideal gas at standard conditions (MFC flows are standard L/min)
MOLAR_VOLUME = 22.414
DEFAULT_REACTOR_VOLUME = 5.0

# Rates (mmol/L/h) below this are within analyzer noise, so ratios using them as the
# denominator are left missing instead of blowing up
MIN_RATE = 1.0


def smooth_gas_signals(df) -> pd.DataFrame:
    """Trailing rolling mean on the gas readings to suppress analyzer noise.

    Trailing (not centered) so a live prediction only uses data already recorded.
    """
    df = df.copy()
    for col in GAS_COLS:
        rolled = df.groupby("batch")[col].rolling(SMOOTH_WINDOW, min_periods=1)
        df[col] = rolled.mean().reset_index(level=0, drop=True)
    return df


def outlet_flow(df):
    """Outlet gas flow (L/min) from an inert (N2) balance: all N2 that enters leaves.

    Consumed CH4 and O2 shrink the gas volume, so outlet flow is lower than inlet flow.
    """
    inert_in = 100 - df["O2_in"] - df["CH4_in"]
    inert_out = 100 - df["O2_out"] - df["CH4_out"] - df["CO2_out"]
    return df["flo"] * inert_in / inert_out


def safe_ratio(numerator, denominator):
    """Ratio of two rates, missing where the denominator is too small to trust."""
    return numerator / denominator.where(denominator > MIN_RATE)


def add_physiological_features(df, reactor_volume) -> pd.DataFrame:
    """Gas uptake/evolution rates (mmol per L of broth per hour) and their ratios.

    OUR: O2 uptake rate, MUR: CH4 uptake rate, CER: CO2 evolution rate,
    RQ: respiratory quotient (CER/OUR), O2_CH4: O2 used per CH4 (OUR/MUR),
    carbon_recovery: fraction of consumed CH4 carbon released as CO2 (CER/MUR).
    Gas concentrations are in %, flow in standard L/min, inlet CO2 is assumed zero.
    """
    df = df.copy()
    # (L/min) x (%) -> mmol/L/h
    to_mmol_per_l_h = 60 * 1000 / (100 * MOLAR_VOLUME * reactor_volume)
    flow_in = df["flo"]
    flow_out = outlet_flow(df)

    df["OUR"] = (flow_in * df["O2_in"] - flow_out * df["O2_out"]) * to_mmol_per_l_h
    df["MUR"] = (flow_in * df["CH4_in"] - flow_out * df["CH4_out"]) * to_mmol_per_l_h
    df["CER"] = flow_out * df["CO2_out"] * to_mmol_per_l_h
    df["RQ"] = safe_ratio(df["CER"], df["OUR"])
    df["O2_CH4"] = safe_ratio(df["OUR"], df["MUR"])
    df["carbon_recovery"] = safe_ratio(df["CER"], df["MUR"])

    return df


def add_trajectory_features(df) -> pd.DataFrame:
    """Lagged values and deltas capturing how each rate is trending."""
    df = df.copy()
    for feat in PHYSIO_FEATURES:
        for lag in LAG_INTERVALS:
            df[f"{feat}_lag{lag}"] = df.groupby("batch")[feat].shift(lag)
            df[f"{feat}_delta{lag}"] = df[feat] - df[f"{feat}_lag{lag}"]
    return df


def add_cumulative_features(df) -> pd.DataFrame:
    """Integrated gas consumption/production since batch start (mass-balance backbone).

    Rows are 1 minute apart, so summing mmol/L/h rates and dividing by 60 gives mmol/L.
    """
    df = df.copy()
    for feat in CUMULATIVE_FEATURES:
        df[f"{feat}_cumsum"] = df.groupby("batch")[feat].cumsum() / 60

    batch_start = df.groupby("batch")["timestamp"].transform("first")
    df["time_since_batch_start"] = (
        df["timestamp"] - batch_start
    ).dt.total_seconds() / 60
    return df


def add_rolling_features(df) -> pd.DataFrame:
    """Trailing rolling mean and standard deviation of each rate."""
    df = df.copy()
    for feat in PHYSIO_FEATURES:
        for window in ROLLING_WINDOWS:
            rolled = df.groupby("batch")[feat].rolling(window, min_periods=1)
            df[f"{feat}_rollmean{window}"] = rolled.mean().reset_index(
                level=0, drop=True
            )
            df[f"{feat}_rollstd{window}"] = rolled.std().reset_index(level=0, drop=True)
    return df


def add_batch_phase_feature(df) -> pd.DataFrame:
    """Fraction of the batch elapsed (0 at start, 1 at the latest row of the batch)."""
    df = df.copy()
    batch_duration = df.groupby("batch")["time_since_batch_start"].transform("max")
    df["batch_phase"] = df["time_since_batch_start"] / batch_duration.replace(0, pd.NA)
    return df


def build_features(merged_df, reactor_volume) -> pd.DataFrame:
    """Full feature pipeline for one run's merged sensor + manual data."""
    df = smooth_gas_signals(merged_df)
    df = add_physiological_features(df, reactor_volume)
    df = add_trajectory_features(df)
    df = add_cumulative_features(df)
    df = add_rolling_features(df)
    df = add_batch_phase_feature(df)
    return df


def build_run_features(
    workbook_path, reactor_volume, clip_to_manual=True
) -> pd.DataFrame:
    """Load one run workbook and return its feature table tagged with run_id."""
    run_id, bioflo, gas, manual = load_workbook(workbook_path)
    merged = merge_sources(bioflo, gas, manual, clip_to_manual=clip_to_manual)
    features = build_features(merged, reactor_volume)
    features["run_id"] = run_id
    return features


def build_feature_table(workbook_paths, reactor_volume) -> pd.DataFrame:
    """Feature tables for several runs stacked into one DataFrame."""
    run_dfs = []
    for path in workbook_paths:
        run_dfs.append(build_run_features(path, reactor_volume))
    return pd.concat(run_dfs, ignore_index=True)


def get_feature_columns(df):
    """All numeric columns except labels and identifiers."""
    feature_cols = []
    for col in df.select_dtypes(include="number").columns:
        if col not in NON_FEATURE_COLS:
            feature_cols.append(col)
    return feature_cols
