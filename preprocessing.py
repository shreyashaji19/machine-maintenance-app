"""Data loading, cleaning and feature engineering (shared by training AND prediction)."""
import numpy as np
import pandas as pd

TARGET = "Machine failure"
TYPE_MAP = {"L": 0, "M": 1, "H": 2}
# IDs + failure-reason columns. The reasons are only known AFTER a failure -> data leakage.
DROP_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Temp_diff", "Power"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60  # watts
    return df


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {TARGET, "Type", "Air temperature [K]"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df["Type"] = df["Type"].map(TYPE_MAP)
    df = add_features(df)
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]


def row_from_input(d: dict) -> pd.DataFrame:
    """Turn one web-form / API input into a model-ready single-row DataFrame."""
    row = pd.DataFrame([{
        "Type": TYPE_MAP[d["type"]],
        "Air temperature [K]": float(d["air_temp"]),
        "Process temperature [K]": float(d["process_temp"]),
        "Rotational speed [rpm]": float(d["rpm"]),
        "Torque [Nm]": float(d["torque"]),
        "Tool wear [min]": float(d["tool_wear"]),
    }])
    return add_features(row)[FEATURES]
