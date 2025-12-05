import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone


def _to_numpy_strict(X_df, y_ser, fill_value=0):
    X = X_df.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(fill_value)

    y = pd.to_numeric(y_ser, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = y.fillna(y.median() if fill_value == "median" else fill_value)

    X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
    y_np = np.ascontiguousarray(y.to_numpy(dtype=np.float32).ravel())
    return X_np, y_np


def _mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


# -----------------------------
# ONLY 10-fold replica function
# -----------------------------
def run_kfold_replica_once(
    df,
    descriptor_cols,
    base_model,
    best_params,
    solubility_col="solubility_g_100g_log",
    n_splits=10,
    seed=0,
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmse_list, mae_list, r2_list = [], [], []

    for train_idx, test_idx in kf.split(df):
        X_train_df = df.iloc[train_idx][descriptor_cols]
        y_train_ser = df.iloc[train_idx][solubility_col]
        X_test_df  = df.iloc[test_idx][descriptor_cols]
        y_test_ser = df.iloc[test_idx][solubility_col]

        X_train, y_train = _to_numpy_strict(X_train_df, y_train_ser)
        X_test,  y_test  = _to_numpy_strict(X_test_df,  y_test_ser)

        model = clone(base_model)
        if best_params:
            model.set_params(**best_params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_list.append(mean_absolute_error(y_test, y_pred))
        r2_list.append(r2_score(y_test, y_pred))

    return {
        "rmse": float(np.mean(rmse_list)),
        "mae": float(np.mean(mae_list)),
        "r2":  float(np.mean(r2_list)),
    }


# -----------------------------
# MAIN: REPLICATED 10-FOLD ONLY
# -----------------------------
def run_replicated_cv(
    df,
    descriptor_cols,
    solute_name_col,     # kept for signature consistency, not used
    base_model,
    model_tag,
    best_params_kfold=None,
    best_params_loso=None,  # unused but kept to maintain API
    rs_n_jobs=-1,
    n_rep=20,
    base_seed=123,
    output_dir=".",
    solubility_col="solubility_g_100g_log",
):

    # pick tuned params (first from list)
    if best_params_kfold:
        best_params_rep = best_params_kfold[0]
    else:
        best_params_rep = {}

    kfold_rmse, kfold_mae, kfold_r2 = [], [], []

    print(f"[Replica] Running {n_rep} replicated 10-fold CV runs…")
    for r in range(n_rep):
        seed = base_seed + r
        metrics = run_kfold_replica_once(
            df=df,
            descriptor_cols=descriptor_cols,
            base_model=base_model,
            best_params=best_params_rep,
            solubility_col=solubility_col,
            n_splits=10,
            seed=seed,
        )
        kfold_rmse.append(metrics["rmse"])
        kfold_mae.append(metrics["mae"])
        kfold_r2.append(metrics["r2"])

    # compute statistics
    rmse_mean, rmse_std = _mean_std(kfold_rmse)
    mae_mean, mae_std   = _mean_std(kfold_mae)
    r2_mean,  r2_std    = _mean_std(kfold_r2)

    uncertainty_df = pd.DataFrame([{
        "model_tag": model_tag,
        "cv_type": "10-fold",
        "RMSE_mean": rmse_mean,
        "RMSE_std":  rmse_std,
        "MAE_mean":  mae_mean,
        "MAE_std":   mae_std,
        "R2_mean":   r2_mean,
        "R2_std":    r2_std,
    }])

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"{model_tag}_uncertainty_summary.csv")
    uncertainty_df.to_csv(outpath, index=False)

    print("[Replica] Saved:", outpath)
    print(uncertainty_df)

    return uncertainty_df
