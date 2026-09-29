"""Loading run workbooks and aligning sensor streams onto a common time grid."""

import os

import pandas as pd


def run_id_from_path(path):
    """Workbooks are named '<RUN_ID>_DATA.xlsx', e.g. 'RUN01_DATA.xlsx'."""
    return os.path.basename(path).split("_")[0]


def load_workbook(path):
    """Read the Bioflo (process), Gas (off-gas analyzer) and Manual (OD samples) sheets."""
    run_id = run_id_from_path(path)
    bioflo = pd.read_excel(path, sheet_name=f"{run_id} Bioflo")
    gas = pd.read_excel(path, sheet_name=f"{run_id} Gas")
    manual = pd.read_excel(path, sheet_name=f"{run_id} Manual")
    return run_id, bioflo, gas, manual


def merge_sources(bioflo, gas, manual, clip_to_manual=True) -> pd.DataFrame:
    """Align all sources on a 1-minute grid.

    With clip_to_manual=True (training) the data is clipped to the time range covered
    by every sheet. With clip_to_manual=False (live prediction) the end of the range is
    set by the sensor streams only, so the latest sensor reading is kept even when no
    manual OD sample has been taken since.
    """
    bioflo = bioflo.copy()
    gas = gas.copy()
    manual = manual.copy()
    bioflo["timestamp"] = bioflo["timestamp"].dt.round("min")
    gas["timestamp"] = gas["timestamp"].dt.round("min")
    manual["timestamp"] = manual["timestamp"].dt.round("min")

    start = max(
        bioflo["timestamp"].min(), gas["timestamp"].min(), manual["timestamp"].min()
    )
    end = min(bioflo["timestamp"].max(), gas["timestamp"].max())
    if clip_to_manual:
        end = min(end, manual["timestamp"].max())

    bioflo = bioflo[
        (bioflo["timestamp"] >= start) & (bioflo["timestamp"] <= end)
    ].set_index("timestamp")
    gas = gas[(gas["timestamp"] >= start) & (gas["timestamp"] <= end)].set_index(
        "timestamp"
    )
    manual = manual[
        (manual["timestamp"] >= start) & (manual["timestamp"] <= end)
    ].set_index("timestamp")

    # Interpolate the 10-minute gas and 1-minute bioflo data onto a shared 1-minute grid
    # so every manual OD timestamp lines up with a sensor row
    bioflo = bioflo.resample("min").mean().interpolate()
    gas = gas.resample("min").mean().interpolate()
    merged = bioflo.join(gas, how="outer")

    # Add the manual data and fill batch numbers across the sensor rows
    merged = merged.join(manual, how="outer")
    merged["batch"] = merged["batch"].bfill().ffill()

    return merged.reset_index()
