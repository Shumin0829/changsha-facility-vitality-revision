"""Check saved predictions, folds and optionally trusted locally fitted models."""
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import clone

import reanalysis as r
from schema import read_main, read_extended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--extended-input", type=Path)
    parser.add_argument("--facility-input", type=Path)
    parser.add_argument("--trust-local-models", action="store_true", help="Load only models you created locally; joblib/pickle can execute code")
    parser.add_argument("--refit", action="store_true", help="Optional two XGBoost refits; not needed for lightweight verification")
    args = parser.parse_args()
    if args.refit and not args.trust_local_models:
        parser.error("--refit requires --trust-local-models")
    out = args.out
    df = read_main(args.input, args.encoding)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    supplied = {"main_input": args.input, "extended_input": args.extended_input, "facility_input": args.facility_input}
    for name, path in supplied.items():
        if path is not None and r.digest(path) != manifest["sources"][name]["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
    full_frame = read_extended(args.extended_input, df, args.encoding) if args.extended_input else None
    full = full_frame.drop(columns=["x", "y", "等级", "SV_weekday", "SV_weekend"]) if full_frame is not None else None
    projected = np.column_stack(r.Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True).transform(df.x.to_numpy(), df.y.to_numpy()))
    checked = []
    checkpoints = sorted((out / "checkpoints").glob("*.json"))
    if not checkpoints:
        raise ValueError("No completed checkpoints to verify")
    for file in checkpoints:
        summary = json.loads(file.read_text(encoding="utf-8"))
        tag, target, model = summary["protocol"], summary["target"], summary["model"]
        mapping = np.arange(len(df))
        if tag.endswith("central5"):
            if args.facility_input is None:
                raise ValueError("--facility-input needed to verify central-subset provenance")
            mapping = pd.read_csv(out / "central_subset_rows.csv")["source_row_id"].to_numpy() - 1
        current, coords = df.iloc[mapping].reset_index(drop=True), projected[mapping]
        if tag.endswith("full26") or tag.endswith("vif_trainfold"):
            if full is None:
                raise ValueError("--extended-input needed for 26-feature verification")
            X = full
        else:
            X = current[r.FEATURES]
        predictions = pd.read_csv(out / "predictions" / f"{tag}_{target}_{model}.csv")
        assert len(predictions) == len(current)
        assert np.array_equal(predictions.row_id.to_numpy(), np.arange(len(current)) + 1)
        assert np.isfinite(predictions[["prediction", "residual"]].to_numpy()).all()
        assert np.allclose(predictions[target], current[target], rtol=0, atol=1e-8)
        assert np.allclose(predictions.residual, predictions[target] - predictions.prediction, rtol=0, atol=1e-8)
        for metric, value in r.metric(predictions[target], predictions.prediction).items():
            assert abs(value - summary["OOF_" + metric]) < 1e-8
        membership = pd.read_csv(out / "splits" / (tag + "_outer_membership.csv"))
        split_details = json.loads((out / "splits" / (tag + "_details.json")).read_text(encoding="utf-8"))
        gap = split_details["buffer_m"]
        seen, maximum_difference = np.zeros(len(current)), None
        for fold in range(1, manifest["config"]["outer_folds"] + 1):
            members = membership[membership.fold == fold]
            assert len(members) == len(current) and members.row_id.nunique() == len(current)
            train = members.loc[members.role == "train", "row_id"].to_numpy() - 1
            test = members.loc[members.role == "test", "row_id"].to_numpy() - 1
            seen[test] += 1
            assert not np.intersect1d(train, test).size
            assert (predictions.set_index("row_id").loc[test + 1, "fold"] == fold).all()
            assert np.allclose(predictions.set_index("row_id").loc[test + 1, "baseline_prediction"], current[target].iloc[train].mean(), rtol=0, atol=1e-8)
            if tag != "random_5fold":
                assert not set(members.loc[members.role == "train", "block"]) & set(members.loc[members.role == "test", "block"])
                assert cKDTree(coords[test]).query(coords[train], k=1)[0].min() >= gap - 1e-6
            if args.trust_local_models:
                fitted = joblib.load(out / "models" / f"{tag}_{target}_{model}_fold{fold}.joblib")
                saved = predictions.set_index("row_id").loc[test + 1, "prediction"].to_numpy()
                difference = float(np.max(np.abs(fitted.predict(X.iloc[test]) - saved)))
                maximum_difference = max(maximum_difference or 0, difference)
                assert difference < 1e-7
        assert (seen == 1).all()
        checked.append({"protocol": tag, "target": target, "model": model, "n": len(current), "prediction_reload_max_abs_diff": maximum_difference, "OOF_metrics_match": True, "fold_coverage_exactly_once": True, "split_checks_pass": True})
    refits = []
    if args.refit:
        tag = "spatial_2km_buffer500"
        members = pd.read_csv(out / "splits" / (tag + "_outer_membership.csv"))
        first = members[members.fold == 1]
        train = first.loc[first.role == "train", "row_id"].to_numpy() - 1
        test = first.loc[first.role == "test", "row_id"].to_numpy() - 1
        for target in ["SV_weekday", "SV_weekend"]:
            fitted = joblib.load(out / "models" / f"{tag}_{target}_XGBoost_fold1.joblib")
            refitted = clone(fitted).fit(df[r.FEATURES].iloc[train], df[target].iloc[train])
            difference = float(np.max(np.abs(refitted.predict(df[r.FEATURES].iloc[test]) - fitted.predict(df[r.FEATURES].iloc[test]))))
            assert difference < 1e-7
            refits.append({"target": target, "outer_fold": 1, "max_abs_prediction_diff": difference})
    r.dump({"checks": checked, "independent_refits": refits, "model_loading_opt_in": args.trust_local_models, "note": "Verifies this reanalysis only; no original-result, measurement, causal or synthetic-to-real validation"}, out / "verification_results.json")
    print("VERIFIED combinations", len(checked), "models reloaded", len(checked) * 5 if args.trust_local_models else 0)


if __name__ == "__main__":
    main()
