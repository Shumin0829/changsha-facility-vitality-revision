"""Synthetic-only tests; does not rerun or validate real study results."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

import reanalysis as r
import plot_revision_figures_1009 as plot
from generate_synthetic import generate
from schema import MAIN_FEATURES, read_main, read_extended, read_facilities, validate_numeric


def expect_rejection(frame):
    try:
        validate_numeric(frame, ["x", "y", "等级", "SV_weekday", "SV_weekend"] + MAIN_FEATURES)
    except ValueError:
        return
    raise AssertionError("Invalid input was not rejected")


def smoke_plot_inputs(root, script_directory):
    examples = root / "plot_examples"
    source, _, _ = generate(examples, n=1650)
    analysis_root = root / "fictional_shap"
    (analysis_root / "shap").mkdir(parents=True)
    source_path = examples / "synthetic_main.csv"
    manifest = {"sources": {"main_input": {"sha256": plot.sha256(source_path)}}}
    (analysis_root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rng = np.random.default_rng(20261009)
    folds = np.resize(np.arange(1, 6), len(source))
    for target in plot.DATES:
        values = pd.DataFrame(rng.normal(size=(len(source), len(MAIN_FEATURES))), columns=MAIN_FEATURES)
        values.insert(0, "row_id", np.arange(1, len(source) + 1))
        values["fold"] = folds
        prefix = f"spatial_2km_buffer500_{target}_XGBoost"
        values.to_csv(analysis_root / "shap" / f"{prefix}_values.csv", index=False)
        importance = values[MAIN_FEATURES].abs().mean().rename("mean_abs_SHAP")
        importance.index.name = "feature"
        importance.to_csv(analysis_root / "shap" / f"{prefix}_importance.csv")
    data, values, importances, provenance = plot.load_sources(analysis_root, source_path)
    assert len(data) == 1650 and set(values) == set(plot.DATES)
    assert all(np.isclose(table.relative_percent.sum(), 100) for table in importances.values())
    assert provenance[0]["input_hash_matches_analysis_manifest"]
    legacy = {"sources": {"0728.csv": {"sha256": plot.sha256(source_path)}}}
    (analysis_root / "run_manifest.json").write_text(json.dumps(legacy), encoding="utf-8")
    _, _, _, legacy_provenance = plot.load_sources(analysis_root, source_path)
    assert legacy_provenance[0]["manifest_input_key"] == "0728.csv"
    assert legacy_provenance[0]["input_hash_matches_analysis_manifest"]
    (analysis_root / "run_manifest.json").unlink()
    try:
        plot.load_sources(analysis_root, source_path)
    except ValueError:
        pass
    else:
        raise AssertionError("Plotting loader silently accepted missing input-hash verification")
    _, _, _, unverified = plot.load_sources(analysis_root, source_path, allow_unverified_input_hash=True)
    assert not unverified[0]["input_hash_matches_analysis_manifest"]
    (analysis_root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    changed_path = examples / "changed.csv"
    source.loc[0, "BD"] += 0.01
    source.to_csv(changed_path, index=False, encoding="utf-8-sig")
    try:
        plot.load_sources(analysis_root, changed_path)
    except ValueError:
        pass
    else:
        raise AssertionError("Plotting loader accepted a changed predictor file")
    subprocess.run([sys.executable, str(script_directory / "plot_revision_figures_1009.py"),
                    "--analysis-root", str(analysis_root), "--input", str(source_path),
                    "--out", str(root / "plot_metadata"), "--metadata-only"],
                   check=True, cwd=script_directory)
    assert (root / "plot_metadata" / "figure_contract.json").is_file()
    return "fictional SHAP row alignment, importance normalization, missing/changed input-hash rejection and explicit legacy override, metadata-only plotting CLI (no rendering or real SHAP)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, help="Optional synthetic validation report path")
    args = parser.parse_args()
    checks = []
    with tempfile.TemporaryDirectory(prefix="synthetic_vitality_") as temporary:
        root = Path(temporary)
        examples, results = root / "examples", root / "results"
        main_frame, _, _ = generate(examples)
        data = read_main(examples / "synthetic_main.csv")
        assert MAIN_FEATURES == r.FEATURES and len(data) == 400
        read_extended(examples / "synthetic_extended26.csv", data)
        read_facilities(examples / "synthetic_facilities.csv", data)
        checks.append("main/extended/facility schemas and row alignment")
        script_directory = Path(__file__).resolve().parent
        for name in ["synthetic_main.csv", "synthetic_extended26.csv", "synthetic_facilities.csv"]:
            generated = pd.read_csv(examples / name, encoding="utf-8-sig")
            packaged = pd.read_csv(script_directory / "examples" / name, encoding="utf-8-sig")
            pd.testing.assert_frame_equal(generated, packaged, check_exact=False, rtol=0, atol=1e-7)
        checks.append("all packaged example tables match deterministic synthetic regeneration")
        invalid = main_frame.drop(columns="SID").copy()
        expect_rejection(invalid)
        invalid = main_frame.copy()
        invalid.loc[0, "BD"] = np.nan
        expect_rejection(invalid)
        invalid = main_frame.copy()
        invalid.loc[0, "FL"] = -1
        expect_rejection(invalid)
        checks.append("missing field, missing value, inconsistent radius rejected")
        alias_file = examples / "alias.csv"
        main_frame.rename(columns={"IRD": "ID", "SFA": "POIL"}).to_csv(alias_file, index=False, encoding="utf-8-sig")
        read_main(alias_file)
        checks.append("unambiguous named aliases accepted")
        coordinates = np.column_stack(r.Transformer.from_crs(4326, 32649, always_xy=True).transform(data.x, data.y))
        groups = r.block_groups(coordinates, 2000, np.zeros(2))
        outer = r.make_splits(coordinates, groups, True, 5, r.SEED, 500)
        coverage = np.zeros(len(data))
        for fold, (train, test) in enumerate(outer, 1):
            coverage[test] += 1
            r.make_splits(coordinates[train], groups[train], True, 3, r.SEED + fold, 500)
        assert (coverage == 1).all()
        checks.append("nested 5x3 spatial fold coverage and centre exclusion")
        command = [sys.executable, str(script_directory / "reanalysis.py"), "--input", str(examples / "synthetic_main.csv"), "--out", str(results), "--models", "OLS", "--jobs", "1", "--skip-shap", "--stage", "main"]
        subprocess.run(command, check=True, cwd=script_directory)
        subprocess.run([sys.executable, str(script_directory / "verify_results.py"), "--input", str(examples / "synthetic_main.csv"), "--out", str(results), "--trust-local-models"], check=True, cwd=script_directory)
        verification = json.loads((results / "verification_results.json").read_text(encoding="utf-8"))
        assert len(verification["checks"]) == 4
        assert not verification["independent_refits"]
        checks.append("20 synthetic OLS outer fits, pooled metrics, fold coverage and trusted model reload")
        checks.append(smoke_plot_inputs(root, script_directory))
    report = {"synthetic_only": True, "passed": True, "checks": checks, "packaged_examples_match_synthetic_regeneration": True, "real_data_read": False, "real_models_refitted": 0, "scientific_findings_validated": False, "full_180_model_analysis_rerun": False, "plot_rendering_tested": False, "plot_visual_QA_performed": False}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

