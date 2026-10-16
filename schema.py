"""Named-column validation; never silently remove rows or impute values."""
from pathlib import Path

import numpy as np
import pandas as pd


BASE_COLUMNS = ["x", "y", "等级", "SV_weekday", "SV_weekend"]
MAIN_FEATURES = ["BD", "PR", "MD", "ED", "ResiD", "ECED", "SID", "SI", "Mix", "FL", "SFA", "NDVI", "NDWI", "DSS", "DBS", "IRD", "RD", "DCBD", "DRA", "DSF"]
ALIASES = {"ID": "IRD", "POIL": "SFA", "SFR": "SFA"}
EXTENDED_FEATURES = MAIN_FEATURES + ["SD", "ResD", "TD", "PFD", "DID", "POID"]


def canonicalize(frame):
    renamed = {}
    for original, canonical in ALIASES.items():
        if original in frame:
            if canonical in frame or canonical in renamed.values():
                raise ValueError(f"Ambiguous alias for {canonical}; supply one column only")
            renamed[original] = canonical
    return frame.rename(columns=renamed)


def validate_numeric(frame, columns, expected_n=None):
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if expected_n is not None and len(frame) != expected_n:
        raise ValueError(f"Expected {expected_n} rows, found {len(frame)}")
    if len(frame) < 80:
        raise ValueError("At least 80 rows required; spatial folds may require more")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite required values; no rows were removed")
    if not frame["x"].between(-180, 180).all() or not frame["y"].between(-90, 90).all():
        raise ValueError("x/y must be WGS84 longitude/latitude, not projected coordinates")
    if not frame["等级"].isin([1, 2, 3, 4]).all():
        raise ValueError("等级 must be an integer planning level from 1 to 4")
    expected_radius = frame["等级"].map({1: 5000, 2: 2500, 3: 1000, 4: 500})
    if not np.array_equal(frame["FL"].to_numpy(), expected_radius.to_numpy()):
        raise ValueError("FL must encode the level-specific service radius in metres")
    if frame.duplicated(["x", "y"]).any():
        raise ValueError("Duplicate point coordinates need an explicit author decision")
    return frame


def read_main(path, encoding="utf-8-sig", expected_n=None):
    frame = canonicalize(pd.read_csv(Path(path), encoding=encoding))
    return validate_numeric(frame, BASE_COLUMNS + MAIN_FEATURES, expected_n)


def validate_alignment(left, right, atol=1e-7):
    if len(left) != len(right):
        raise ValueError("Input tables have different row counts")
    for column in BASE_COLUMNS:
        if not np.allclose(left[column], right[column], rtol=0, atol=atol):
            raise ValueError(f"Input row alignment failed for {column}")


def read_extended(path, main, encoding="utf-8-sig"):
    frame = canonicalize(pd.read_csv(Path(path), encoding=encoding))
    validate_numeric(frame, BASE_COLUMNS + EXTENDED_FEATURES, len(main))
    actual = [column for column in frame if column not in BASE_COLUMNS]
    if set(actual) != set(EXTENDED_FEATURES) or len(actual) != 26:
        raise ValueError("Extended sensitivity requires exactly the documented 26 features")
    validate_alignment(main, frame)
    if not np.allclose(main[MAIN_FEATURES], frame[MAIN_FEATURES], rtol=0, atol=1e-7):
        raise ValueError("Extended input changed a primary predictor")
    return frame


def read_facilities(path, main, encoding="utf-8-sig"):
    frame = pd.read_csv(Path(path), encoding=encoding)
    required = ["x", "y", "等级", "Y_点活", "adname"]
    if any(column not in frame for column in required):
        raise ValueError(f"Facility input requires {required}")
    aligned = frame.rename(columns={"Y_点活": "SV_weekday"}).copy()
    aligned["SV_weekend"] = main["SV_weekend"].to_numpy()
    validate_alignment(main, aligned)
    if frame.adname.isna().any():
        raise ValueError("Facility district labels cannot be missing")
    return frame

