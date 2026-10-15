"""Reviewer-requested reanalysis with explicit, user-supplied inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import warnings

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from schema import read_main, read_extended, read_facilities, validate_alignment

FEATURES = ["BD", "PR", "MD", "ED", "ResiD", "ECED", "SID", "SI", "Mix", "FL", "SFA", "NDVI", "NDWI", "DSS", "DBS", "IRD", "RD", "DCBD", "DRA", "DSF"]
MAPPING = {"BD": "BD", "PR": "FAR", "MD": "MFD", "ED": "ED", "ResiD": "ResiD", "ECED": "ESCFD", "SID": "SFD", "SI": "POISI", "Mix": "SFSI", "FL": "SFL_radius_encoded", "SFA": "SFA", "NDVI": "NDVI", "NDWI": "NDWI", "DSS": "DSS", "DBS": "DBS", "IRD": "ID", "RD": "RD", "DCBD": "DCBD", "DRA": "DRA", "DSF": "DSF"}
OUTER_K = 5
INNER_K = 3
SEED = 7
JOBS = 4
BUDGET = 12
LOG_PATH = None
SHAP_ENABLED = True


def csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def dump(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)), encoding="utf-8")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def log(text):
    line = time.strftime("%H:%M:%S") + " " + text
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def vif_values(X):
    X = np.asarray(X, dtype=float)
    Z = (X - np.mean(X, axis=0)) / np.maximum(np.std(X, axis=0), 1e-12)
    result = []
    for i in range(Z.shape[1]):
        others = np.delete(Z, i, axis=1)
        beta = np.linalg.lstsq(others, Z[:, i], rcond=None)[0]
        ratio = np.sum((Z[:, i] - others @ beta)**2) / max(np.sum(Z[:, i]**2), 1e-12)
        result.append(1.0 / max(ratio, 1e-12))
    return np.asarray(result)


class VIFScreen(BaseEstimator, TransformerMixin):
    """Exploratory VIF selection learned in each training fold only."""
    def __init__(self, threshold=5.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.keep_ = np.arange(X.shape[1])
        while len(self.keep_) > 1:
            values = vif_values(np.asarray(X)[:, self.keep_])
            if values.max() <= self.threshold:
                break
            self.keep_ = np.delete(self.keep_, np.argmax(values))
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


def models():
    spaces = {
        "OLS": {},
        "Ridge": {"model__alpha": np.logspace(-3, 4, 12).tolist()},
        "RandomForest": {"model__n_estimators": [150, 300], "model__max_depth": [None, 5, 10, 20], "model__min_samples_leaf": [1, 3, 8], "model__max_features": [0.6, 1.0]},
        "GBDT": {"model__n_estimators": [100, 300, 600], "model__learning_rate": [0.03, 0.06, 0.1], "model__max_depth": [2, 3, 4], "model__min_samples_leaf": [3, 10], "model__subsample": [0.8, 1.0]},
        "XGBoost": {"model__n_estimators": [200, 500, 1000], "model__learning_rate": [0.03, 0.06, 0.1], "model__max_depth": [2, 3, 5, 7], "model__min_child_weight": [1, 5, 10], "model__subsample": [0.8, 1.0], "model__colsample_bytree": [0.8, 1.0], "model__reg_lambda": [1.0, 7.0, 20.0]},
        "LGBM": {"model__n_estimators": [200, 500, 1000], "model__learning_rate": [0.03, 0.06, 0.1], "model__num_leaves": [7, 15, 31], "model__min_child_samples": [10, 20, 40], "model__colsample_bytree": [0.8, 1.0], "model__reg_lambda": [0.0, 7.0, 20.0]},
    }
    estimators = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(random_state=SEED, n_jobs=1),
        "GBDT": GradientBoostingRegressor(random_state=SEED, loss="squared_error"),
        "XGBoost": XGBRegressor(random_state=SEED, n_jobs=1, objective="reg:squarederror", tree_method="hist", verbosity=0),
        "LGBM": LGBMRegressor(random_state=SEED, n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True),
    }
    result = {}
    for name, estimator in estimators.items():
        steps = [("scale", StandardScaler())] if name in ("OLS", "Ridge") else []
        steps.append(("model", estimator))
        result[name] = (Pipeline(steps), spaces[name])
    return result


def block_groups(coords, width, origin):
    grid = np.floor((coords - origin) / width).astype(int)
    _, groups = np.unique(grid, axis=0, return_inverse=True)
    return groups


def make_splits(coords, groups, spatial, n_splits, seed, buffer_m=0):
    base = GroupKFold(n_splits=n_splits).split(coords, groups=groups) if spatial else KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(coords)
    splits = []
    for train, test in base:
        if spatial and buffer_m:
            distance, _ = cKDTree(coords[test]).query(coords[train], k=1)
            train = train[distance >= buffer_m]
        assert len(train) >= 40 and len(test) >= 5, (len(train), len(test), buffer_m)
        assert not np.intersect1d(train, test).size
        if spatial:
            assert not np.intersect1d(groups[train], groups[test]).size
            if buffer_m:
                assert cKDTree(coords[test]).query(coords[train], k=1)[0].min() >= buffer_m - 1e-8
        splits.append((train, test))
    return splits


def metric(y, pred):
    return {"R2": float(r2_score(y, pred)), "RMSE": float(np.sqrt(mean_squared_error(y, pred))), "MAE": float(mean_absolute_error(y, pred))}


def fit_search(est, space, X, y, cv):
    if not space:
        est = clone(est).fit(X, y)
        return est, {}, None, pd.DataFrame()
    search = RandomizedSearchCV(clone(est), space, n_iter=BUDGET, scoring="neg_root_mean_squared_error", cv=cv, random_state=SEED, n_jobs=JOBS, pre_dispatch=JOBS, error_score="raise", return_train_score=False)
    with joblib.parallel_backend("loky", inner_max_num_threads=1):
        search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(-search.best_score_), pd.DataFrame(search.cv_results_)


def run_protocol(out, tag, X, df, coords, width=2000, buffer_m=500, names=None, vif=False):
    spatial = tag != "random_5fold"
    groups = block_groups(coords, width, np.zeros(2))
    outer = make_splits(coords, groups, spatial, OUTER_K, SEED, buffer_m)
    modelspec = models()
    names = list(modelspec) if names is None else names
    splits_records = []
    for fold, (tr, te) in enumerate(outer, 1):
        closest = cKDTree(coords[tr]).query(coords[te], k=1)[0]
        splits_records.append({"fold": fold, "train_n": len(tr), "test_n": len(te), "excluded_n": len(df)-len(tr)-len(te), "min_nearest_train_m": float(closest.min()), "median_nearest_train_m": float(np.median(closest))})
    dump({"protocol": tag, "block_width_m": width if spatial else None, "buffer_m": buffer_m if spatial else 0, "folds": splits_records}, out/"splits"/(tag+"_details.json"))
    # Membership for exact reproduction; omitted rows are buffer exclusions.
    membership = []
    for fold, (tr, te) in enumerate(outer, 1):
        roles = np.full(len(df), "buffer_excluded", dtype=object)
        roles[tr], roles[te] = "train", "test"
        membership.extend({"row_id": i+1, "fold": fold, "role": roles[i], "block": int(groups[i]), "x": df.iloc[i]["x"], "y": df.iloc[i]["y"]} for i in range(len(df)))
    csv(pd.DataFrame(membership), out/"splits"/(tag+"_outer_membership.csv"))
    for target in ("SV_weekday", "SV_weekend"):
        for name in names:
            runkey = f"{tag}_{target}_{name}"
            marker = out/"checkpoints"/(runkey+".json")
            if marker.exists():
                log("CHECKPOINT " + runkey)
                continue
            log("START " + runkey)
            rows = []
            pred_all = np.full(len(df), np.nan)
            base_all = np.full(len(df), np.nan)
            shap_all = np.full(X.shape, np.nan)
            shap_by_fold = []
            fold_id = np.full(len(df), -1)
            for fold, (tr, te) in enumerate(outer, 1):
                inner = make_splits(coords[tr], groups[tr], spatial, INNER_K, SEED+fold, buffer_m)
                est, space = modelspec[name]
                if vif:
                    est = Pipeline([("vif", VIFScreen()), ("model", clone(est.named_steps["model"]))])
                start = time.monotonic()
                fitted, params, inner_rmse, history = fit_search(est, space, X.iloc[tr], df[target].iloc[tr], inner)
                if len(history):
                    csv(history, out/"search"/f"{runkey}_fold{fold}.csv")
                dump(params, out/"parameters"/f"{runkey}_fold{fold}.json")
                predict = fitted.predict(X.iloc[te])
                baseline = np.full(len(te), df[target].iloc[tr].mean())
                baseline_rmse = metric(df[target].iloc[te], baseline)["RMSE"]
                rec = {"protocol": tag, "target": target, "model": name, "fold": fold, "train_n": len(tr), "test_n": len(te), "feature_n": len(fitted.named_steps["vif"].keep_) if vif else X.shape[1], "inner_RMSE": inner_rmse, "fit_seconds": time.monotonic()-start, "baseline_RMSE": baseline_rmse, **metric(df[target].iloc[te], predict)}
                pred_all[te], base_all[te], fold_id[te] = predict, baseline, fold
                rows.append(rec)
                if SHAP_ENABLED and name == "XGBoost" and tag in ("random_5fold", "spatial_2km_buffer500"):
                    import shap
                    explainer = shap.TreeExplainer(fitted.named_steps["model"], feature_perturbation="tree_path_dependent")
                    values = explainer.shap_values(X.iloc[te])
                    error = np.max(np.abs(values.sum(axis=1) + explainer.expected_value - predict))
                    assert error < 0.01, error
                    shap_all[te] = values
                    importance = np.abs(values).mean(axis=0)
                    shap_by_fold.extend({"fold": fold, "feature": column, "mean_abs_SHAP": float(value), "relative_importance": float(value / importance.sum()), "rank": float(rank)} for column,value,rank in zip(X.columns,importance,pd.Series(-importance).rank(method="min")))
                if vif:
                    dump({"kept": X.columns[fitted.named_steps["vif"].keep_].tolist()}, out/"parameters"/f"{runkey}_fold{fold}_features.json")
                # Save outer-fit models for transparent review, not future-independent validation.
                joblib.dump(fitted, out/"models"/f"{runkey}_fold{fold}.joblib", compress=3)
                csv(pd.DataFrame(rows), out/"metrics"/(runkey+"_folds.csv"))
                log(f"FOLD {fold} {runkey}: R2={rec['R2']:.4f} RMSE={rec['RMSE']:.3f}")
            assert np.isfinite(pred_all).all() and (fold_id > 0).all()
            predictions = df[["x", "y", target]].copy()
            predictions.insert(0, "row_id", np.arange(len(df))+1)
            predictions["fold"], predictions["prediction"], predictions["baseline_prediction"] = fold_id, pred_all, base_all
            predictions["residual"] = df[target] - pred_all
            csv(predictions, out/"predictions"/(runkey+".csv"))
            if np.isfinite(shap_all).all():
                values = pd.DataFrame(shap_all, columns=X.columns)
                values.insert(0, "row_id", np.arange(len(df))+1)
                values["fold"] = fold_id
                csv(values, out/"shap"/(runkey+"_values.csv"))
                csv(pd.DataFrame(shap_by_fold), out/"shap"/(runkey+"_fold_importance.csv"))
                importance = np.abs(shap_all).mean(axis=0)
                imp = pd.DataFrame({"feature": X.columns, "manuscript_label": [MAPPING.get(c,c) for c in X.columns], "mean_abs_SHAP": importance, "relative_importance": importance/importance.sum()}).sort_values("mean_abs_SHAP", ascending=False)
                imp["rank"] = np.arange(len(imp))+1
                csv(imp, out/"shap"/(runkey+"_importance.csv"))
            summary = {"protocol": tag, "target": target, "model": name, "n": len(df), "p": X.shape[1], "OOF_" + "R2": metric(df[target],pred_all)["R2"], "OOF_RMSE": metric(df[target],pred_all)["RMSE"], "OOF_MAE": metric(df[target],pred_all)["MAE"], "OOF_baseline_RMSE": metric(df[target],base_all)["RMSE"]}
            for m in ("R2", "RMSE", "MAE"):
                summary[m+"_fold_mean"] = float(np.mean([r[m] for r in rows]))
                summary[m+"_fold_SD"] = float(np.std([r[m] for r in rows], ddof=1))
            dump(summary, marker)
            log("DONE " + runkey)


def moran(value, coords, k, seed=7, permutations=999):
    n = len(value)
    distances, neigh = cKDTree(coords).query(coords, k=k+1)
    W = csr_matrix((np.ones(n*k), (np.repeat(np.arange(n), k), neigh[:,1:].ravel())),shape=(n,n))
    W = W.maximum(W.T)
    W = W.multiply(1 / np.asarray(W.sum(axis=1)))
    z = np.asarray(value)-np.mean(value)
    denom = z @ z
    observed = float(n / W.sum() * (z @ (W @ z)) / denom)
    rng = np.random.default_rng(seed)
    simulations = np.empty(permutations)
    for i in range(permutations):
        perm = rng.permutation(z)
        simulations[i] = n/W.sum()*(perm @ (W @ perm))/denom
    expected = -1/(n-1)
    p = (1+np.sum(np.abs(simulations-expected) >= abs(observed-expected)))/(permutations+1)
    return {"I": observed, "expected_I": expected, "permutations": permutations, "p_two_sided": float(p), "k": k, "median_kth_distance_m": float(np.median(distances[:,-1]))}


def audit(out, df):
    audit_data = {"n": len(df), "csv_level_counts": df["等级"].value_counts().to_dict(), "p_main": len(FEATURES), "duplicate_xy": int(df.duplicated(["x", "y"]).sum()), "missing_required": int(df[FEATURES].isna().sum().sum()), "feature_mapping": MAPPING, "FL_vs_level": pd.crosstab(df["等级"], df["FL"]).to_dict(), "interpretation": "Input-schema audit, not verification of upstream geospatial measurement or causal conclusions"}
    dump(audit_data,out/"data_audit.json")
    csv(pd.DataFrame({"csv_column":FEATURES,"manuscript_label":[MAPPING[c] for c in FEATURES],"mean":df[FEATURES].mean().values,"SD":df[FEATURES].std().values,"VIF_descriptive_only":vif_values(df[FEATURES])}),out/"variable_dictionary.csv")
    df[FEATURES].corr(method="spearman").to_csv(out/"spearman_features.csv",encoding="utf-8-sig")


def summarize(out,df,coords):
    records=[json.loads(f.read_text(encoding="utf-8")) for f in (out/"checkpoints").glob("*.json")]
    if not records:
        raise ValueError("No completed checkpoints available to summarize")
    csv(pd.DataFrame(records).sort_values(["protocol","target","model"]),out/"model_comparison.csv")
    stat=[]
    for target in ("SV_weekday","SV_weekend"):
        variables={"outcome":df[target]}
        for tag in ("random_5fold","spatial_2km_buffer500"):
            for name in models():
                p=out/"predictions"/f"{tag}_{target}_{name}.csv"
                if p.exists():
                    variables[tag+"_"+name+"_OOF_residual"]=pd.read_csv(p)["residual"]
        for var,value in variables.items():
            ks=(4,8,16) if var=="outcome" or "XGBoost" in var else (8,)
            for k in ks:
                stat.append({"target":target,"variable":var,**moran(value,coords,k)})
    stats=pd.DataFrame(stat)
    p=stats.p_two_sided.to_numpy()
    order=np.argsort(p); q=np.empty(len(p))
    q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    stats["BH_FDR_q"]=q
    csv(stats,out/"moran_diagnostics.csv")
    correlations=[]
    for target in ("SV_weekday","SV_weekend"):
        path_a=out/"shap"/f"random_5fold_{target}_XGBoost_importance.csv"
        path_b=out/"shap"/f"spatial_2km_buffer500_{target}_XGBoost_importance.csv"
        if not path_a.exists() or not path_b.exists():
            continue
        a=pd.read_csv(path_a).set_index("feature")
        b=pd.read_csv(path_b).set_index("feature")
        rho=spearmanr(a.loc[FEATURES,"rank"],b.loc[FEATURES,"rank"]).statistic
        A=set(a.sort_values("rank").head(5).index); B=set(b.sort_values("rank").head(5).index)
        correlations.append({"target":target,"rank_Spearman_random_vs_spatial":rho,"top5_intersection_n":len(A&B),"random_top5":", ".join(a.sort_values("rank").head(5).index),"spatial_top5":", ".join(b.sort_values("rank").head(5).index)})
    csv(pd.DataFrame(correlations),out/"shap_stability.csv")


def main():
    global LOG_PATH, JOBS, BUDGET, SHAP_ENABLED
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True,help="Main table with the documented 20 predictors")
    ap.add_argument("--out",type=Path,required=True,help="Private, versioned output directory")
    ap.add_argument("--encoding",default="utf-8-sig")
    ap.add_argument("--extended-input",type=Path,help="26-feature exploratory input; required for sensitivity/all")
    ap.add_argument("--facility-input",type=Path,help="Aligned district inventory; required for sensitivity/all")
    ap.add_argument("--facility-encoding",default="utf-8-sig")
    ap.add_argument("--expected-n",type=int,help="Use 1650 to enforce the manuscript analysis sample size")
    ap.add_argument("--models",nargs="+",choices=list(models()),default=list(models()))
    ap.add_argument("--jobs",type=int,default=JOBS)
    ap.add_argument("--search-budget",type=int,default=BUDGET)
    ap.add_argument("--skip-shap",action="store_true",help="Testing only; the reported main analysis includes SHAP")
    ap.add_argument("--central-districts",nargs="+",default=["芙蓉区","天心区","雨花区","岳麓区","开福区"])
    ap.add_argument("--stage",choices=["main","sensitivity","summary","all"],default="main")
    args=ap.parse_args(); out=args.out
    if args.jobs < 1 or args.search_budget < 1:
        ap.error("jobs and search-budget must be positive")
    if args.stage in ("sensitivity", "all") and (args.extended_input is None or args.facility_input is None):
        ap.error("sensitivity/all requires --extended-input and --facility-input")
    JOBS, BUDGET, SHAP_ENABLED = args.jobs, args.search_budget, not args.skip_shap
    df=read_main(args.input,args.encoding,args.expected_n)
    extended=read_extended(args.extended_input,df,args.encoding) if args.extended_input else None
    facilities=read_facilities(args.facility_input,df,args.facility_encoding) if args.facility_input else None
    sources={"main_input":{"sha256":digest(args.input)}}
    if args.extended_input:
        sources["extended_input"]={"sha256":digest(args.extended_input),"predictor_order":[c for c in extended if c not in ["x","y","等级","SV_weekday","SV_weekend"]]}
    if args.facility_input:
        sources["facility_input"]={"sha256":digest(args.facility_input)}
    config={"outer_folds":OUTER_K,"inner_folds":INNER_K,"search_candidates_per_tuned_model":BUDGET,"seed":SEED,"jobs":JOBS,"models":args.models,"SHAP":SHAP_ENABLED,"central_districts":args.central_districts,"spatial_primary":{"block_m":2000,"buffer_m":500},"script_sha256":digest(Path(__file__))}
    old_manifest=out/"run_manifest.json"
    if old_manifest.exists():
        previous=json.loads(old_manifest.read_text(encoding="utf-8"))
        if previous["config"] != config:
            raise ValueError("Run configuration changed; choose a fresh output directory")
        for label,info in sources.items():
            if label in previous["sources"] and previous["sources"][label] != info:
                raise ValueError("Input changed; choose a fresh output directory")
        sources={**previous["sources"],**sources}
    for directory in ("checkpoints","metrics","search","parameters","predictions","splits","models","shap"):
        (out/directory).mkdir(parents=True,exist_ok=True)
    LOG_PATH = out / "analysis_log.txt"
    environment={name:importlib.metadata.version(name) for name in ("numpy","pandas","scipy","scikit-learn","xgboost","lightgbm","shap","pyproj","joblib")}
    environment.update(python=platform.python_version(),platform=platform.platform())
    dump(environment,out/"environment.json")
    dump({"sources":sources,"config":config,"score":"inner mean RMSE minimized; outer R2/RMSE/MAE","early_stopping":False,"status":"Code execution is not upstream measurement validation, causal evidence, or public deposition"},out/"run_manifest.json")
    audit(out,df)
    project=Transformer.from_crs("EPSG:4326","EPSG:32649",always_xy=True)
    coords=np.column_stack(project.transform(df.x.to_numpy(),df.y.to_numpy()))
    # This projection is specific to the study region, not a global distance CRS.
    X=df[FEATURES]
    dump({name:space for name,(_,space) in models().items()},out/"search_spaces.json")
    csv(pd.DataFrame({"row_id":np.arange(len(df))+1,"easting_m":coords[:,0],"northing_m":coords[:,1]}),out/"projected_coordinates.csv")
    started=time.monotonic()
    if args.stage in ("main","all"):
        run_protocol(out,"random_5fold",X,df,coords,buffer_m=0,names=args.models)
        run_protocol(out,"spatial_2km_buffer500",X,df,coords,names=args.models)
    if args.stage in ("sensitivity","all"):
        for tag,width,buffer in [("spatial_1km_buffer500",1000,500),("spatial_4km_buffer500",4000,500),("spatial_2km_buffer1000",2000,1000)]:
            run_protocol(out,tag,X,df,coords,width,buffer,names=["XGBoost"])
        full=extended.drop(columns=["x","y","等级","SV_weekday","SV_weekend"])
        run_protocol(out,"spatial_2km_full26",full,df,coords,names=["XGBoost"])
        run_protocol(out,"spatial_2km_vif_trainfold",full,df,coords,names=["XGBoost"],vif=True)
        central=facilities.adname.isin(args.central_districts).to_numpy()
        if not central.any():
            raise ValueError("No facility rows match --central-districts")
        dump({"sha256":digest(args.facility_input),"n":len(facilities),"district_counts":facilities.adname.value_counts().to_dict(),"central_five_n":int(central.sum()),"central_level_counts":facilities.loc[central,"等级"].value_counts().to_dict(),"row_alignment_atol":1e-7},out/"facility_scope_audit.json")
        subset=df.loc[central].reset_index(drop=True)
        csv(pd.DataFrame({"subset_row_id":np.arange(len(subset))+1,"source_row_id":np.flatnonzero(central)+1}),out/"central_subset_rows.csv")
        run_protocol(out,"spatial_2km_central5",subset[FEATURES],subset,coords[central],names=["XGBoost"])
    if args.stage in ("summary","all"):
        summarize(out,df,coords)
    dump({"stage":args.stage,"completed":True,"duration_seconds":time.monotonic()-started},out/("completed_"+args.stage+".json"))
    log("COMPLETED " + args.stage)


if __name__=="__main__":
    main()
