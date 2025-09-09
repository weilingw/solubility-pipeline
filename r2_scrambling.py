import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # must be before importing pyplot
import matplotlib.pyplot as plt


from joblib import load
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV

# ========= Metrics =========
def _r2_general(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)

# ========= Estimator builders =========
def _clone_with_params(base_model, params: dict | None):
    est = clone(base_model)
    if params:
        est.set_params(**params)
    return est

def _make_estimator(base_model, param_grid, use_random_search, random_state=42):
    est = clone(base_model)
    if use_random_search:
        return RandomizedSearchCV(
            est,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            scoring="neg_root_mean_squared_error",
            random_state=random_state,
            n_jobs=1,
        )
    return est

# ========= (Optional) fixed-param loaders =========
def _extract_fixed_params_from_estimator(fitted_estimator, model_type: str):
    """Keep only the knobs that matter; tolerate bare estimators or pipelines."""
    p = fitted_estimator.get_params(deep=True)
    if model_type.lower() == "xgb":
        keep = ["n_estimators","max_depth","learning_rate","subsample",
                "colsample_bytree","reg_alpha","reg_lambda","min_child_weight",
                "gamma","tree_method","device","n_jobs","random_state","predictor"]
    elif model_type.lower() == "rf":
        keep = ["n_estimators","max_depth","min_samples_leaf","max_features",
                "min_samples_split","bootstrap","random_state","n_jobs"]
    elif model_type.lower() == "svm":
        # pipeline keys; scaler uses defaults; freeze SVR hyperparams
        keep = ["svr__kernel","svr__C","svr__epsilon","svr__gamma"]
    else:
        keep = list(p.keys())
    return {k: p[k] for k in keep if k in p}

def _load_fixed_params_kfold(output_dir: str, tag: str, model_type: str):
    """Return list[dict] or None. Looks in output_dir and its parent."""
    if not output_dir or not tag:
        return None
    # search both scramble_dir and base_output_path
    search_dirs = [output_dir, os.path.dirname(output_dir)]

    # JSON first
    for d in search_dirs:
        json_path = os.path.join(d, f"{tag}_kfold_best_params.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                params = json.load(f)
            if isinstance(params, list) and all(isinstance(dd, dict) for dd in params):
                print(f" Loaded fixed kfold params from JSON: {json_path}")
                return params

    # Then joblib
    for d in search_dirs:
        joblib_path = os.path.join(d, f"{tag}_kfold_models.joblib")
        if os.path.exists(joblib_path):
            models = load(joblib_path)  # expect list
            if isinstance(models, (list, tuple)):
                out = [_extract_fixed_params_from_estimator(m, model_type) for m in models]
                print(f" Extracted fixed kfold params from joblib: {joblib_path}")
                return out
    return None


def _load_fixed_params_loso(output_dir: str, tag: str, model_type: str, solute_names=None):
    """Return dict[str->dict] or None. Looks in output_dir and its parent."""
    if not output_dir or not tag:
        return None
    search_dirs = [output_dir, os.path.dirname(output_dir)]

    # JSON first
    for d in search_dirs:
        json_path = os.path.join(d, f"{tag}_loso_best_params.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                params = json.load(f)
            if isinstance(params, dict):
                print(f" Loaded fixed LOSO params from JSON: {json_path}")
                return params

    # Then joblib
    for d in search_dirs:
        joblib_path = os.path.join(d, f"{tag}_loso_models.joblib")
        if os.path.exists(joblib_path):
            models = load(joblib_path)  # dict[name->est] or list
            if isinstance(models, dict):
                out = {k: _extract_fixed_params_from_estimator(v, model_type) for k, v in models.items()}
                print(f" Extracted fixed LOSO params from joblib map: {joblib_path}")
                return out
            if isinstance(models, (list, tuple)) and solute_names is not None and len(models) == len(solute_names):
                out = {name: _extract_fixed_params_from_estimator(m, model_type)
                       for name, m in zip(solute_names, models)}
                print(f" Extracted fixed LOSO params from joblib list: {joblib_path}")
                return out
    return None


# ========= Plot helper =========
def _save_histogram(perm_R2, obs_R2, title, out_png):
   
    fig, ax = plt.subplots(figsize=(3.3, 2.5))   # single-column safe size

    counts, bins, patches = ax.hist(perm_R2, bins='auto')
    ax.axvline(obs_R2, linewidth=2, color="black")

    xmin = float(min(np.min(perm_R2), obs_R2))
    xmax = float(max(np.max(perm_R2), obs_R2))
    pad  = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)

    y_max = float(np.max(counts)) if counts.size else 1.0
    ax.set_ylim(0, y_max * 1.15 + 1)

    ax.set_title(title)
    ax.set_xlabel("R² under permutation")
    ax.set_ylabel("Count")
    ax.text(0.98, 0.95, f"obs R² = {obs_R2:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ========= Single run helpers (correctly permute **train y only**) =========
def _run_loso_once(df, descriptor_cols, solute_name_col, target_col,
                   base_model, param_grid, use_random_search, model_type,
                   fixed_params_map=None, permute_train_y=False, rng=None):
    y_true_all, y_pred_all = [], []

    # deterministic order for reproducibility across runs
    solutes = sorted(df[solute_name_col].unique().tolist())
    for sol in solutes:
        train_df = df[df[solute_name_col] != sol]
        test_df  = df[df[solute_name_col] == sol]

        X_train = train_df[descriptor_cols]
        y_train = train_df[target_col].to_numpy()
        X_test  = test_df[descriptor_cols]
        y_test  = test_df[target_col].to_numpy()

        if model_type.lower() == "svm":
            X_train = X_train.fillna(0); X_test = X_test.fillna(0)

        # Permute TRAIN labels only
        if permute_train_y:
            assert rng is not None, "rng required when permute_train_y=True"
            y_fit = rng.permutation(y_train)
        else:
            y_fit = y_train

        if fixed_params_map and sol in fixed_params_map:
            model = _clone_with_params(base_model, fixed_params_map[sol])
        else:
            model = _make_estimator(base_model, param_grid, use_random_search)

        model.fit(X_train, y_fit)
        fitted = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        y_pred = fitted.predict(X_test)

        y_true_all.extend(y_test.tolist()); y_pred_all.extend(y_pred.tolist())

    y_true_all = np.asarray(y_true_all); y_pred_all = np.asarray(y_pred_all)
    return dict(
        R2   = _r2_general(y_true_all, y_pred_all),
        RMSE = float(np.sqrt(np.mean((y_pred_all - y_true_all)**2))),
        MAE  = float(np.mean(np.abs(y_pred_all - y_true_all))),
    )

def _run_kfold_once(df, descriptor_cols, kfold_splits, target_col,
                    base_model, param_grid, use_random_search, model_type,
                    fixed_params_by_fold=None, permute_train_y=False, rng=None):
    y_true_all, y_pred_all = [], []

    for fold_id, (train_idx, test_idx) in enumerate(kfold_splits):
        X_train = df.iloc[train_idx][descriptor_cols]
        y_train = df.iloc[train_idx][target_col].to_numpy()
        X_test  = df.iloc[test_idx][descriptor_cols]
        y_test  = df.iloc[test_idx][target_col].to_numpy()

        if model_type.lower() == "svm":
            X_train = X_train.fillna(0); X_test = X_test.fillna(0)

        # Permute TRAIN labels only
        if permute_train_y:
            assert rng is not None, "rng required when permute_train_y=True"
            y_fit = rng.permutation(y_train)
        else:
            y_fit = y_train

        if fixed_params_by_fold and fold_id < len(fixed_params_by_fold) and fixed_params_by_fold[fold_id]:
            model = _clone_with_params(base_model, fixed_params_by_fold[fold_id])
        else:
            model = _make_estimator(base_model, param_grid, use_random_search)

        model.fit(X_train, y_fit)
        fitted = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        y_pred = fitted.predict(X_test)

        y_true_all.extend(y_test.tolist()); y_pred_all.extend(y_pred.tolist())

    y_true_all = np.asarray(y_true_all); y_pred_all = np.asarray(y_pred_all)
    return dict(
        R2   = _r2_general(y_true_all, y_pred_all),
        RMSE = float(np.sqrt(np.mean((y_pred_all - y_true_all)**2))),
        MAE  = float(np.mean(np.abs(y_pred_all - y_true_all))),
    )

# ========= Public APIs =========
def permutation_test_loso(
    df, descriptor_cols, solute_name_col,
    base_model, param_grid, model_type, use_random_search,
    B=200, speedy_perms=True, target_col="solubility_g_100g_log",
    random_state=808, output_dir=None, tag=None, make_plot=False
):
    """
    Return (obs_metrics_dict, perm_R2_array, p_value).
    - Uses frozen LOSO (deterministic solute order)
    - Permutes TRAIN labels only, per left-out split
    - If <tag>_loso_best_params.json (or *_loso_models.joblib) exists in output_dir,
      re-fits permutations with those fixed best hyperparameters (no re-search).
    """
    os.makedirs(output_dir or ".", exist_ok=True)

    # Observed
    obs = _run_loso_once(
        df, descriptor_cols, solute_name_col, target_col,
        base_model, param_grid, use_random_search, model_type,
        fixed_params_map=None, permute_train_y=False, rng=None
    )

    # Try to load fixed params
    fixed_map = _load_fixed_params_loso(output_dir, tag, model_type,
                                        solute_names=sorted(df[solute_name_col].unique()))
    # Determine whether to use search in permutation
    use_rs = False if fixed_map is not None else ((not speedy_perms) and use_random_search)

    # Permutations
    perm_R2 = np.zeros(B, dtype=float)
    for b in range(B):
        rng = np.random.default_rng(int(random_state) + b)
        res = _run_loso_once(
            df, descriptor_cols, solute_name_col, target_col,
            base_model, param_grid, use_rs, model_type,
            fixed_params_map=fixed_map, permute_train_y=True, rng=rng
        )
        perm_R2[b] = res["R2"]

    p_val = (1 + np.sum(perm_R2 >= obs["R2"])) / (1 + B)

    # Save
    stem = f"{tag}_" if tag else ""
    if output_dir is not None:
        csv_path = os.path.join(output_dir, f"{stem}loso_r2_scrambling.csv")
        pd.DataFrame({
            "perm_index": np.arange(1, B + 1),
            "R2_perm": perm_R2,
            "obs_R2": obs["R2"],
            "p_value": p_val
        }).to_csv(csv_path, index=False)

        # extras to help downstream plotting / provenance
        np.save(os.path.join(output_dir, f"{stem}loso_perm_r2.npy"), perm_R2)
        with open(os.path.join(output_dir, f"{stem}loso_obs_r2.json"), "w") as f:
            json.dump({"obs_r2": float(obs["R2"])}, f)

        if make_plot:
            _save_histogram(perm_R2, obs["R2"], "R² scrambling (LOSO)",
                            os.path.join(output_dir, f"{stem}loso_r2_scrambling.png"))

    return obs, perm_R2, p_val

def permutation_test_kfold(
    df, descriptor_cols, kfold_splits,
    base_model, param_grid, model_type, use_random_search,
    B=200, speedy_perms=True, target_col="solubility_g_100g_log",
    random_state=808, output_dir=None, tag=None, make_plot=False
):
    """
    Return (obs_metrics_dict, perm_R2_array, p_value).
    - Uses provided frozen 10-fold splits
    - Permutes TRAIN labels only, per fold
    - If <tag>_kfold_best_params.json (or *_kfold_models.joblib) exists in output_dir,
      re-fits permutations with those fixed best hyperparameters (no re-search).
    """
    os.makedirs(output_dir or ".", exist_ok=True)

    # Observed
    obs = _run_kfold_once(
        df, descriptor_cols, kfold_splits, target_col,
        base_model, param_grid, use_random_search, model_type,
        fixed_params_by_fold=None, permute_train_y=False, rng=None
    )

    # Try to load fixed params
    fixed_by_fold = _load_fixed_params_kfold(output_dir, tag, model_type)
    use_rs = False if fixed_by_fold is not None else ((not speedy_perms) and use_random_search)

    # Permutations
    perm_R2 = np.zeros(B, dtype=float)
    for b in range(B):
        rng = np.random.default_rng(int(random_state) + b)
        res = _run_kfold_once(
            df, descriptor_cols, kfold_splits, target_col,
            base_model, param_grid, use_rs, model_type,
            fixed_params_by_fold=fixed_by_fold, permute_train_y=True, rng=rng
        )
        perm_R2[b] = res["R2"]

    p_val = (1 + np.sum(perm_R2 >= obs["R2"])) / (1 + B)

    # Save
    stem = f"{tag}_" if tag else ""
    if output_dir is not None:
        csv_path = os.path.join(output_dir, f"{stem}kfold_r2_scrambling.csv")
        pd.DataFrame({
            "perm_index": np.arange(1, B + 1),
            "R2_perm": perm_R2,
            "obs_R2": obs["R2"],
            "p_value": p_val
        }).to_csv(csv_path, index=False)

        # extras
        np.save(os.path.join(output_dir, f"{stem}kfold_perm_r2.npy"), perm_R2)
        with open(os.path.join(output_dir, f"{stem}kfold_obs_r2.json"), "w") as f:
            json.dump({"obs_r2": float(obs["R2"])}, f)

        if make_plot:
            _save_histogram(perm_R2, obs["R2"], "R² scrambling (10-fold)",
                            os.path.join(output_dir, f"{stem}kfold_r2_scrambling.png"))

    return obs, perm_R2, p_val
