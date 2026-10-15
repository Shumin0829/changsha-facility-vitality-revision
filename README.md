# Changsha Sports Facility Vitality: Reanalysis Code

## Status and Data Boundaries

This repository contains the manuscript-revision reanalysis and Figures 5-11
renderer from the prepared `1009` code package. It is not an untouched copy of the
2025 exploratory notebook, and it does not include upstream GIS/heat-map
extraction or the later GWR/MGWR benchmark experiments. See `RELEASE_NOTES.md`
for this release's scope and publication-only changes.

Repository: https://github.com/Shumin0829/changsha-facility-vitality-revision

No DOI or additional software reuse licence is asserted. This release is not
represented as open-source licensed; see `NOTICE.md`. Real facility coordinates,
heat maps, POI inventories, private messages, original CSV attachments, and fitted
real-data models are not included.

The examples are entirely synthetic. Their 400 generated rows are unrelated to
the observed 1,650 facilities. Smoke-test success demonstrates software plumbing
only, not the paper's scientific findings or model performance.

## Environment

The synthetic test used Python 3.12.14; see the locked environment supplied with
the revision for exact dependency versions. Installation is an explicit user action:

```sh
python -m venv .venv
python -m pip install -r requirements-lock.txt
```

Use the newly created environment's Python after activating it. The lock records
the previously used environment; availability of these versions on every platform
is not guaranteed. No installation is performed by this package. Do not use
`python -O`, because invariant checks deliberately use assertions.

## Input Schema

Supply a CSV explicitly with `--input`; outputs require `--out`. UTF-8 with optional
BOM is the default; use `--encoding gbk` for the author's historical CSVs.
Rows must remain in the same facility order across all provided tables.
The loader rejects missing/non-finite required values and duplicate coordinates;
it never silently drops rows, imputes values, or changes the sample.

| Fields | Meaning |
| --- | --- |
| `x`, `y` | WGS84 longitude and latitude, degrees; distances use UTM 49N, EPSG:32649, specific to the study region |
| `等级` | Planning level 1-4; not a measured sport-activity outcome |
| `SV_weekday`, `SV_weekend` | Heat-map point values on one representative working day and one representative weekend day, respectively |
| `BD`, `PR` | Building density and floor-area-ratio indicators |
| `MD`, `ED`, `ResiD`, `ECED`, `SID` | Selected POI density indicators; manuscript aliases include MFD, ED, ResiD, ESCFD, SFD |
| `SI`, `Mix` | General POI diversity and sports-facility diversity; manuscript aliases POISI and SFSI |
| `FL` | Service-radius encoding in metres: level 1/2/3/4 maps to 5000/2500/1000/500; not an ordinal numeric level |
| `SFA`, `NDVI`, `NDWI` | Facility-area indicator, vegetation index, water index |
| `DSS`, `DBS`, `IRD`, `RD`, `DCBD`, `DRA`, `DSF` | Transport/destination distance and density indicators; retain author's documented units |

The named main predictor list contains exactly 20 fields and is fixed in
`schema.py` and `reanalysis.py`. Other columns are not selected for the primary
model. Supported aliases are `ID -> IRD` and `POIL/SFR -> SFA`; ambiguous aliases
are rejected. This numeric naming correspondence does not independently validate
indicator definitions, GIS extraction, units, entropy formula, or POI taxonomy.
The software does not perform outcome normalization or distance-unit conversion.

The recovered main sample contains 1,650 rows, with level counts
13/40/182/1415. Use `--expected-n 1650` to enforce its sample size. The synthetic
examples intentionally have a different size and distribution. Historical
1,400-record manuscript counts must not be inferred from feature deletion.

The recovered 30-column earlier input had 25 predictors, including `SID`. The
later 25-column main input removed exactly five predictors: `SD`, `ResD`, `TD`,
`PFD`, and `DID`, leaving 20 predictors; it did not remove facilities. A separate
31-column working copy adds `POID`, giving 26 predictors. The optional full-26
and training-fold VIF analyses are expanded exploratory sensitivity analyses,
not strict replication of the original 25-predictor notebook. The legacy
test-set early-stopping replay is intentionally excluded from this release.

Optional `--extended-input` requires the 20 main predictors plus `SD`, `ResD`,
`TD`, `PFD`, `DID`, `POID`, the five base fields, and aligned values. Preserve
the supplied expanded table's predictor order for exact seeded tree reproduction.
Optional `--facility-input` requires aligned `x`, `y`, `等级`, `Y_点活`, and
`adname` fields; `Y_点活` must match `SV_weekday`. This restricted inventory is
not distributed. District labels are used only for the predefined central-five
subset, not automatic reconstruction of the historical 1,400 sample.

## Protocol

The primary reanalysis compares OLS, Ridge, Random Forest, GBDT, XGBoost, and
LightGBM. Five outer folds provide held-out predictions; three inner folds tune
each tuned model using 12 randomized candidate configurations and mean inner
RMSE. Seed is 7. OLS is untuned. Scaling for OLS/Ridge is fitted inside each
training-fold pipeline. No outer-test early stopping or outer-test hyperparameter
selection is performed. Search spaces and fitted parameters are exported.

Random KFold is shuffled. Spatial GroupKFold uses a fixed UTM grid with origin
(0,0), primary block width 2 km, and removes training facility centres within
500 m of held-out facility centres in both outer and inner folds. Sensitivities
use 1/4 km blocks, 1000 m exclusion, expanded 26 predictors, training-fold VIF
screening, and the central-five subset. These exclusion distances are not a
new service-buffer experiment: large service-area polygons can still overlap.

Each facility receives one outer-test prediction per protocol/model/target.
OOF R2/RMSE/MAE are pooled over those predictions; fold mean/SD are also reported.
Cross-fitted XGBoost SHAP uses `tree_path_dependent` and checks additivity.
SHAP is descriptive attribution, not causation, significance, or a behavioural
measurement. Moran diagnostics use symmetric row-standardized kNN weights,
999 permutations, and a pooled BH-FDR correction. One weekday/weekend observation
cannot establish general weekday/weekend behaviour, seasonality, or causality;
heat-map values are not independently validated sport-participation counts.

## Commands

Validate schemas and run lightweight, entirely synthetic OLS smoke tests:

```sh
python smoke_test.py
```

Run the main 20-predictor reanalysis on authorized data:

```sh
python reanalysis.py --input data/main.csv --encoding gbk --out results/main --expected-n 1650 --stage main
python verify_results.py --input data/main.csv --encoding gbk --out results/main
```

All primary and sensitivity analyses require the authorized extra inputs:

```sh
python reanalysis.py --input data/main.csv --extended-input data/expanded26.csv --facility-input data/facilities.csv --encoding gbk --facility-encoding utf-8-sig --out results/all --expected-n 1650 --stage all
python verify_results.py --input data/main.csv --extended-input data/expanded26.csv --facility-input data/facilities.csv --encoding gbk --out results/all
```

Input hashes and configuration guard against resuming a different dataset or
configuration into existing checkpoints. Use a new output directory for changes.
To summarize a completed main-only run, repeat its configuration with
`--stage summary`. `--models OLS --skip-shap --jobs 1` is a testing option,
not the reported six-model analysis.

Model reload is deliberately opt-in: add `--trust-local-models` only for models
you personally generated. Never load untrusted joblib/pickle files. Optional
`--refit` adds two XGBoost refits and is not needed for the smoke test. Output
tables can contain real coordinates and outcomes; keep them private unless
permissions allow sharing. The release ZIP includes no generated real outputs.

## Figures 5-11

`plot_revision_figures_1009.py` reads saved held-out XGBoost SHAP values without
fitting models. It requires explicit `--analysis-root`, `--input`, and `--out`
arguments. The CSV must be the same 1,650-row main input, in the original row
order; its hash must match the saved analysis manifest's `main_input` key, or the
explicit legacy main-table key `0728.csv`. Private source paths in old manifests
are not read or reproduced. For legacy
outputs without such a hash, the loader stops unless the author independently
verifies row identity and explicitly supplies `--allow-unverified-input-hash`.
This override does not establish identity and is recorded as unverified. Relative
importance is recomputed from absolute SHAP values and checked against the
saved importance tables. Neither predictor values nor SHAP values are rescaled.
SFA uses a labelled symlog display axis, and DRA retains its recorded units.

The renderer is manuscript-specific: observation-date captions are fixed to
18 March and 22 March 2025, and the sample size and fold protocol are fixed to
this revision. `--metadata-only` writes proposed captions/contracts, not proof
that inputs were read or that figures were rendered. Review all proposed captions
against the actual provenance. Upstream DRA and SFA unit definitions remain
author checks; the code does not resolve them.

Figures 5-11 were rendered and individually visually checked using the original
figure-generation script in a separate Python 3.12.8 environment recorded in
`render_environment_reference.json`. The public copy retains all 12 plotting
and caption functions unchanged (AST comparison); its input loader and CLI are
adapted. `requirements-plot.txt` pins that separate renderer's direct dependencies,
not a combination tested with the locked model-analysis environment. Create and
activate a separate environment before installing these packages; do not replace
the model environment's NumPy/pandas versions:

```sh
python -m venv .plot-venv
```

After activating `.plot-venv`, install deliberately and render:

```sh
python -m pip install -r requirements-plot.txt
python plot_revision_figures_1009.py --analysis-root results/main --input data/main.csv --encoding gbk --out outputs/figures
```

The script writes SVG, PDF, 600 dpi PNG, a contact sheet, and private source/QA
tables. Those outputs can reveal real observations and must not be automatically
uploaded with the code. The adapted public interface also completed a full
end-to-end render in that separate figure environment using authorized existing
inputs and saved held-out SHAP, without fitting models. All seven public-copy
PNGs were byte- and pixel-identical to the individually visually checked frozen
figures; relative SHAP shares, point counts, input hashes, and the unchanged 12
plotting functions were verified. Those real render outputs and source observations
remain in a private QA directory, not this release. This was not a rerun of model
analysis and does not resolve outstanding upstream unit definitions.

## Provenance and Release Checklist

The model, search-space, metric, VIF, spatial-block and split functions are
adapted from the locally executed reviewer-requested reanalysis. Changes provide
explicit inputs, schema checks, no real-data copying, hash-safe resume, and safe
opt-in model loading. The packaged model-analysis interface has been smoke-tested
on synthetic data; the plotting interface additionally passed the end-to-end check
above on existing authorized saved results. Neither test reran the 180 real-data
outer fitted models. Existing revision
results belong to their separately recorded execution environment and inputs.

Real data and generated outputs remain excluded from this public code repository.
The historical validation JSON files describe their recorded tests, not a fresh
rerun of the paper's real-data analysis or an archival DOI. Additional reuse
licensing and upstream indicator-unit checks remain separate author decisions.
`build_release.py` only builds and checks a local archive; it does not upload or
verify a remote publication.

