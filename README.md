# Anonymous Review Code

This source-only snapshot provides the manuscript-revision regression model
comparison, random and spatially buffered nested cross-validation, XGBoost SHAP
calculation, and result-verification routines. The model implementations include
OLS, ridge regression, random forest, GBDT, XGBoost and LightGBM.

## Scope

No research data, synthetic data, coordinates, images, fitted model files,
prediction outputs, notebooks or private correspondence are distributed.
Inputs must be supplied separately by an authorized user. The source cannot
reproduce the manuscript's numerical results without those restricted inputs.
This snapshot does not include the upstream GIS extraction, figure rendering,
the historical exploratory notebook, or later GWR/MGWR experiments. It must not
be described as the complete source code for all historical analyses.

## Files

- `reanalysis.py`: model definitions, tuning, cross-validation, SHAP and diagnostics.
- `schema.py`: explicit named-column input validation and alignment checks.
- `verify_results.py`: checks of privately generated predictions and saved fits.
- `requirements.txt`: pinned analysis dependencies.
- `requirements-lock.txt`: recorded historical environment.

## Execution

The recorded environment used Python 3.12.14. Installation is an explicit user
action. Dependency availability can differ across platforms.

```sh
python -m venv .venv
python -m pip install -r requirements-lock.txt
python reanalysis.py --help
python reanalysis.py --input /path/to/authorized-input.csv --out /path/to/private-results --encoding utf-8-sig --stage main
python reanalysis.py --input /path/to/authorized-input.csv --out /path/to/private-results --encoding utf-8-sig --stage summary
python verify_results.py --input /path/to/authorized-input.csv --out /path/to/private-results
```

Activate the virtual environment before executing the commands. The paths above
are placeholders, not included files. Use `--encoding gbk` only for inputs with
that actual encoding. The schema is defined in `schema.py`; aliases are explicit,
missing values are not imputed, and rows are not silently dropped. The main
specification has 20 predictors and two separately modeled observation outcomes.
Predictor and outcome values are not normalized by this interface.

The primary protocol uses five outer folds, three inner folds, 12 randomized
candidate configurations, seed 7, 2-km spatial blocks and a 500-m train/test
exclusion distance. These are algorithm settings, not distributed observations.
The distance projection is region-specific (EPSG:32649); do not assume it is
suitable for another geographic region. No historical distance-unit conversion
is guessed by the software. SHAP explains model predictions, not causal effects.

Optional sensitivity analyses require the documented extra authorized inputs.
For `--stage sensitivity` or `--stage all`, explicitly supply the five relevant
district labels using `--central-districts`; this anonymous copy does not embed
place-name defaults. This CLI adaptation does not change the primary analysis
or its algorithms. It must not be represented as an unchanged historical file.

Outputs can contain restricted input-derived information. Keep them private and
outside the source directory; do not upload them to the review mirror. The
software does not send data to a remote service. Loading joblib/pickle models
can execute code; `--trust-local-models` is only for fits you created and trust.

## Availability And Permissions

See `NOTICE.md`. Public inspection is not an additional open-source reuse licence.
