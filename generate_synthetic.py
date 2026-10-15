"""Create fictional test tables. No real source data is read."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from pyproj import Transformer

from schema import BASE_COLUMNS, MAIN_FEATURES


def generate(directory, n=400):
    if n < 400:
        raise ValueError("Use at least 400 synthetic rows for spatial smoke tests")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261009)
    side = int(np.ceil(np.sqrt(n)))
    east, north = np.meshgrid(np.arange(side), np.arange(side))
    metric = np.column_stack([east.ravel()[:n], north.ravel()[:n]]) * 1600.0
    metric += np.array([450000.0, 3300000.0]) + rng.uniform(-100, 100, (n, 2))
    longitude, latitude = Transformer.from_crs(32649, 4326, always_xy=True).transform(metric[:, 0], metric[:, 1])
    level = np.resize(np.array([1, 2, 3, 4]), n)
    rng.shuffle(level)
    predictors = {name: rng.gamma(2.0, 20.0, n) for name in MAIN_FEATURES}
    predictors["BD"] = rng.uniform(0.02, 0.65, n)
    predictors["PR"] = rng.uniform(0.1, 3.0, n)
    predictors["NDVI"] = rng.uniform(-0.1, 0.8, n)
    predictors["NDWI"] = rng.uniform(-0.4, 0.4, n)
    predictors["SI"] = rng.uniform(0, 2.2, n)
    predictors["Mix"] = rng.uniform(0, 1.4, n)
    predictors["FL"] = pd.Series(level).map({1: 5000, 2: 2500, 3: 1000, 4: 500}).to_numpy()
    for name in ["DSS", "DBS", "DCBD", "DRA", "DSF"]:
        predictors[name] = rng.uniform(10, 5000, n)
    predictors["SFA"] = rng.uniform(200, 5000, n)
    weekday = 80 + 0.4 * predictors["SID"] + 12 * predictors["NDVI"] + rng.normal(0, 5, n)
    weekend = 100 + 0.7 * predictors["SID"] + 8 * predictors["NDWI"] + rng.normal(0, 8, n)
    main = pd.DataFrame({"x": longitude, "y": latitude, "等级": level, "SV_weekday": weekday, "SV_weekend": weekend, **predictors})
    extended = main.copy()
    for name in ["SD", "ResD", "TD", "PFD", "DID", "POID"]:
        extended[name] = rng.gamma(2.0, 20.0, n)
    facilities = main[["x", "y", "等级"]].copy()
    facilities["Y_点活"] = main.SV_weekday
    facilities["adname"] = np.resize(["synthetic_a", "synthetic_b", "synthetic_c", "synthetic_d", "synthetic_e", "synthetic_f"], n)
    main.to_csv(directory / "synthetic_main.csv", encoding="utf-8-sig", index=False)
    extended.to_csv(directory / "synthetic_extended26.csv", encoding="utf-8-sig", index=False)
    facilities.to_csv(directory / "synthetic_facilities.csv", encoding="utf-8-sig", index=False)
    (directory / "SYNTHETIC_NOTICE.json").write_text(json.dumps({"synthetic": True, "n": n, "seed": 20261009, "real_data_read": False, "base_fields": BASE_COLUMNS, "main_features": MAIN_FEATURES, "purpose": "Schema and software smoke tests only; no evidence for real-world scientific claims"}, indent=2), encoding="utf-8")
    return main, extended, facilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("examples"))
    parser.add_argument("--rows", type=int, default=400)
    args = parser.parse_args()
    generate(args.out, args.rows)
    print("Generated fictional data only:", args.rows, "rows")

