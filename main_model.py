# main.py (updated with automatic model metadata handling and visualization)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import re
import json
import numpy as np
import pandas as pd
from joblib import load, dump
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata as ud
from tqdm import tqdm
os.environ["OMP_NUM_THREADS"] = "8"
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from mordred import Calculator, descriptors
from r2_scrambling import permutation_test_kfold, permutation_test_loso
from sklearn.model_selection import KFold 


# === Global Settings ===
model_type = 'rf'              # 'rf', 'xgb', 'svm'
descriptor_type = 'morgan'         # 'morgan', 'mordred', 'moe', 'rdkit'
use_hybrid_mode = True       # Use COSMO features
use_random_search = True     # Whether using 
use_bit_visualization = False      # only for morgan
use_saved_models = False
enable_y_scrambling =  False   # set False to skip all scrambling work

# --- Metadata strings ---
hybrid_str = 'hybrid' if use_hybrid_mode else 'pure'
tuned_str = 'tuned' if use_random_search else 'untuned'
tag = f"{model_type}_{descriptor_type}_{hybrid_str}_{tuned_str}"

input_file = "C:/Project/Nottingham/Tony/solubility/Systematic_study/final_filtered_descriptors.txt"
#input_file = "C:/Project/Solubility_Paper/data_com/cosmotherm.txt"
base_output_path = "C:/Project/Nottingham/Tony/solubility/Systematic_study/checked/main"

# === Load Data ===
df = pd.read_csv(input_file, sep="\t", encoding="utf-8")
df.columns = [re.sub(r'[\u00A0\u200B\u200C\u200D\uFEFF]', ' ', col) for col in df.columns]
df.columns = [ud.normalize('NFKD', col).encode('ascii', 'ignore').decode().strip() for col in df.columns]
df = df.sample(frac=1, random_state=808).reset_index(drop=True)

solute_name_col = next(col for col in df.columns if 'solute' in col.lower() and 'name' in col.lower())
smiles_col = next(col for col in df.columns if 'smiles' in col.lower())

# === Handle COSMO Feature (Hybrid Mode) ===
if use_hybrid_mode:
    cosmo_cols = [col for col in df.columns if 'cosmo' in col.lower() and 'log' in col.lower()]
    if not cosmo_cols:
        raise ValueError("No COSMO log feature found for hybrid mode.")
    cosmo_feature = cosmo_cols[0]
else:
    cosmo_feature = None

if use_hybrid_mode and cosmo_feature:
    cosmo_df = df[[solute_name_col, 'solvent_name', 'solubility_g_100g_log', cosmo_feature]].copy()
    cosmo_df = cosmo_df.rename(columns={cosmo_feature: 'prediction'})
    cosmo_df['model'] = "COSMO-RS (Baseline)"
    cosmo_df['cv_type'] = 'COSMO'
    cosmo_df['solute_name'] = df[solute_name_col]
    cosmo_df = cosmo_df[['solute_name', 'solubility_g_100g_log', 'prediction', 'model', 'cv_type']]
else:
    cosmo_df = pd.DataFrame(columns=['solute_name', 'solubility_g_100g_log', 'prediction', 'model', 'cv_type'])


def compute_mordred_descriptors(smiles_list):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_raw = calc.pandas(mols, nproc=1)  # or: calc.pandas(mols, nproc=1)
    return mordred_raw

# === Generate Descriptors ===
# Remove previously attached solute_ or solvent_ descriptors (MOE excluded)
# Preserve essential metadata before dropping any descriptors
base_cols = df[['solute_name', 'solvent_name', 'solute_smiles', 'solvent_smiles', 'solubility_g_100g_log']].copy()

if descriptor_type in ['morgan', 'mordred', 'rdkit']:
    # Drop *all* solute_/solvent_ descriptors (from MOE or earlier runs)
    protected_cols = {'solute_name', 'solvent_name', 'solute_smiles', 'solvent_smiles'}
    drop_cols = [col for col in df.columns
                 if (col.startswith("solute_") or col.startswith("solvent_")) and col not in protected_cols]
    df = df.drop(columns=drop_cols, errors="ignore")
# Canonicalize SMILES early for consistency
df["solute_smiles"] = df["solute_smiles"].map(
    lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if pd.notnull(x) else x
)
df["solvent_smiles"] = df["solvent_smiles"].map(
    lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if pd.notnull(x) else x
)
if descriptor_type == 'morgan':
    FINGERPRINT_SIZE = 2048
    MORGAN_RADIUS = 2

    def smiles_to_fp_with_bitinfo(smi):
        if not isinstance(smi, str) or not smi.strip():
            return np.zeros(FINGERPRINT_SIZE, dtype=np.int8), {}
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.zeros(FINGERPRINT_SIZE, dtype=np.int8), {}
        bitInfo = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=MORGAN_RADIUS, nBits=FINGERPRINT_SIZE, bitInfo=bitInfo
        )
        arr = np.frombuffer(fp.ToBitString().encode('ascii'), 'S1').view(np.uint8) - ord(b'0')  # fast
        return arr.astype(np.int8, copy=False), bitInfo

    descriptor_cols = []
    bit_mapping = {}
    all_bitInfo = {"solute": {}, "solvent": {}}

    for role, smiles_col in [("solute", "solute_smiles"), ("solvent", "solvent_smiles")]:
        fp_list = []
        role_bitInfo = {}

        # build aligned lists (no skipping)
        for idx, smi in df[smiles_col].items():
            fp_bits, bitInfo = smiles_to_fp_with_bitinfo(smi)
            fp_list.append(fp_bits)
            role_bitInfo[idx] = bitInfo

        fp_array = np.vstack(fp_list)  # shape (n_rows, 2048)
        descriptor_cols_role = [f"{role}_FP_{i}" for i in range(FINGERPRINT_SIZE)]
        df_fp = pd.DataFrame(fp_array, columns=descriptor_cols_role, index=df.index)

        # Shannon entropy filter
        p = df_fp.mean(axis=0)  # fraction of 1s
        entropies = -(p*np.log2(p).where(p.between(1e-12, 1-1e-12), 1) +
                      (1-p)*np.log2(1-p).where(p.between(1e-12, 1-1e-12), 1)).fillna(0.0)

        retained_bits = entropies[entropies > 0.001].index.tolist()
        if retained_bits:
            df = pd.concat([df, df_fp[retained_bits]], axis=1)
            descriptor_cols.extend(retained_bits)
            for col in retained_bits:
                local_bit_id = int(col.replace(f"{role}_FP_", ""))
                bit_mapping[col] = (role, local_bit_id)

        all_bitInfo[role] = role_bitInfo

elif descriptor_type == 'mordred':
    descriptor_cols = []
    all_bitInfo = {"solute": {}, "solvent": {}}

    for role, smiles_col in [('solute', 'solute_smiles'), ('solvent', 'solvent_smiles')]:
        print(f" Computing Mordred descriptors for: {role}")

        mordred_raw = compute_mordred_descriptors(df[smiles_col])
        mordred_raw.index = df.index

        # Step 1: Drop high-NaN columns
        missing_ratio = mordred_raw.isna().mean()
        retained_cols = missing_ratio[missing_ratio <= 0.1].index.tolist()
        mordred_filtered = mordred_raw[retained_cols].fillna(0)

        print(f" {role} descriptors after NaN filtering: {len(retained_cols)}")

        # Step 2: Remove zero-variance descriptors
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.0)
        selector.fit(mordred_filtered)
        retained_final_cols = [mordred_filtered.columns[i] for i in selector.get_support(indices=True)]

        print(f" {role} descriptors after variance filter: {len(retained_final_cols)}")

        ## Step 3: Prefix and append
        mordred_prefixed = mordred_filtered[retained_final_cols].copy()
        mordred_prefixed.columns = [f"{role}_{col}" for col in retained_final_cols]
        df = pd.concat([df, mordred_prefixed], axis=1)

        # Append to descriptor list
        descriptor_cols.extend(mordred_prefixed.columns)

    # Coerce all descriptors to numeric (fix object dtype issues)
    df[descriptor_cols] = df[descriptor_cols].apply(pd.to_numeric, errors='coerce')

    # Final filter to keep only numeric descriptor columns
    descriptor_cols = [col for col in descriptor_cols if np.issubdtype(df[col].dtype, np.number)]


    print(f" Final Mordred descriptor count: {len(descriptor_cols)}")

elif descriptor_type == 'moe':
    print(" Preprocessing MOE descriptors...")

    # Step 0: Drop non-descriptor columns 
    cols_to_exclude = list(range(0, 20)) 
    ## cols_to_exclude = list(range(0, 22)) ##for opencosmo test only 
    all_columns = df.columns.tolist()

    # Initial MOE descriptor selection (excluding known metadata)
    moe_descriptor_cols = [col for i, col in enumerate(all_columns)
                           if i not in cols_to_exclude and np.issubdtype(df[col].dtype, np.number)]

    # Step 1: Drop high-NaN columns (>10%)
    missing_ratio = df[moe_descriptor_cols].isna().mean()
    retained_cols = missing_ratio[missing_ratio <= 0.1].index.tolist()
    df_filtered = df[retained_cols].copy()

    print(f" MOE descriptors after NaN filter: {len(retained_cols)}")

    # Step 2: Fill remaining NaNs with 0
    df_filtered = df_filtered.fillna(0)

    # Step 3: Remove zero-variance columns
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(df_filtered)
    retained_final_cols = [df_filtered.columns[i] for i in selector.get_support(indices=True)]

    print(f" MOE descriptors after variance filter: {len(retained_final_cols)}")

    # Step 4: Coerce to numeric (if necessary) and update df
    df[retained_final_cols] = df_filtered[retained_final_cols].apply(pd.to_numeric, errors='coerce')
    descriptor_cols = retained_final_cols



elif descriptor_type == 'rdkit':
    from rdkit.Chem import Descriptors
    descriptor_names, functions = zip(*Descriptors.descList)
    descriptor_cols = []

    for role, smiles_col in [('solute', 'solute_smiles'), ('solvent', 'solvent_smiles')]:
        rdkit_df = pd.DataFrame(columns=descriptor_names, index=df.index)

        for idx, smi in df[smiles_col].items():
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    values = [func(mol) for func in functions]
                except:
                    values = [None] * len(functions)
            else:
                values = [None] * len(functions)
            rdkit_df.loc[idx] = values

        rdkit_df = rdkit_df.astype(float)
        missing_ratio = rdkit_df.isna().mean()
        retained_cols = missing_ratio[missing_ratio <= 0.1].index.tolist()
        rdkit_df = rdkit_df[retained_cols].fillna(0)

        selector = VarianceThreshold(threshold=0.0)
        selector.fit(rdkit_df)
        retained_final_cols = [rdkit_df.columns[i] for i in selector.get_support(indices=True)]

        rdkit_df_prefixed = rdkit_df[retained_final_cols].copy()
        rdkit_df_prefixed.columns = [f"{role}_{col}" for col in retained_final_cols]
        df = pd.concat([df, rdkit_df_prefixed], axis=1)
        descriptor_cols.extend(rdkit_df_prefixed.columns)

#  Sanity check: restore required columns
for col in ['solute_name', 'solvent_name', 'solubility_g_100g_log']:
    if col not in df.columns:
        df[col] = base_cols[col]

# === Handle COSMO feature for hybrid mode ===
if use_hybrid_mode and cosmo_feature:
    # make sure it is numeric
    df[cosmo_feature] = pd.to_numeric(df[cosmo_feature], errors="coerce")
    if cosmo_feature not in descriptor_cols:
        descriptor_cols.append(cosmo_feature)

# Final filter: only keep numeric descriptors
descriptor_cols = [col for col in descriptor_cols if np.issubdtype(df[col].dtype, np.number)]
print(f"Descriptors used: {len(descriptor_cols)}")


# === Setup ML Model & Hyperparams (GPU/CPU safe) ===
if model_type == 'rf':
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [300, 600, 1000, 1500, 2000],
        "max_depth": [None, 20, 40, 80],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.1, 0.2, 0.3],
        "bootstrap": [True],
        "max_samples": [None, 0.5, 0.75, 0.9],
        "ccp_alpha": [0.0, 1e-4, 1e-3],
    }
    rs_n_jobs = -1  # parallel CV fits on CPU are fine

elif model_type == 'xgb':
    import xgboost as xgb
    ver_major = int(xgb.__version__.split('.')[0].split('-')[0])

    xgb_kwargs = dict(
        objective='reg:squarederror',
        random_state=42,
        tree_method='hist',     # ok for both; 1.x will be overwritten for GPU path below
        eval_metric='rmse',     # optional: keeps logs tidy
        device="cpu",
    )

    if ver_major >= 2:
        # Modern API: use 'device'
        xgb_kwargs.update(device='cpu', n_jobs=8)
        rs_n_jobs = 1
    else:
        # Backward-compatible path for XGBoost 1.x
        xgb_kwargs.update(predictor='cpu_predictor', n_jobs=8)
        rs_n_jobs = 1
    base_model = XGBRegressor(**xgb_kwargs)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }

elif model_type == 'svm':
    base_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])
    param_grid = {
        'svr__C': [0.1, 1, 10, 100],
        'svr__gamma': ['scale', 'auto'],
        'svr__epsilon': [0.01, 0.1, 0.2],
        'svr__kernel': ['rbf']
    }
    rs_n_jobs = -1  # CPU-only; parallel CV fits are fine

else:
    raise ValueError("Invalid model type. Choose 'rf', 'xgb', or 'svm'.")

print(f"Ready to run {model_type.upper()} using {descriptor_type.upper()} descriptors.")
print(f"Descriptors used: {len(descriptor_cols)}")
print(f"COSMO feature included: {cosmo_feature if use_hybrid_mode else 'None'}")


def _to_numpy_strict(X_df, y_ser, fill_value=0):
    X = X_df.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(fill_value)

    y = pd.to_numeric(y_ser, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = y.fillna(y.median() if fill_value == "median" else fill_value)

    X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
    y_np = np.ascontiguousarray(y.to_numpy(dtype=np.float32).ravel())
    return X_np, y_np

prediction_dir = os.path.join(base_output_path, "predictions")
os.makedirs(prediction_dir, exist_ok=True)
# === Train and Save 10-fold Models + Predictions ===
from combined_y_scrambling_plot import build_scrambling_matrices
# Freeze 10-fold splits once (needed for both training and scrambling)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
kfold_splits = list(kf.split(df))  # <- define regardless of use_saved_models
best_params_kfold = []  # NEW

if use_saved_models:
    model_store_kfold = load(os.path.join(base_output_path, f"{tag}_kfold_models.joblib"))
    # INSERT AFTER loading model_store_kfold
    kfold_splits_path = os.path.join(base_output_path, f"{tag}_kfold_test_indices.json")
    if os.path.exists(kfold_splits_path):
        with open(kfold_splits_path, "r", encoding="utf-8") as f:
            kfold_test_indices = json.load(f)
        # sanity: keep length consistent with models
        if len(kfold_test_indices) != len(model_store_kfold):
            print(" Saved k-fold indices count doesn't match number of models. Rebuilding splits with fixed seed.")
            kfold_test_indices = [test_idx.tolist() for (_, test_idx) in kfold_splits]
    else:
        # fallback: rebuild with same RNG; will match if df order unchanged
        print(" No saved k-fold indices found; rebuilding with fixed seed.")
        kfold_test_indices = [test_idx.tolist() for (_, test_idx) in kfold_splits]

    bp_json = os.path.join(base_output_path, f"{tag}_kfold_best_params.json")
    if not os.path.exists(bp_json):
        best_params_kfold = []
        for m in model_store_kfold:
            if hasattr(m, "best_params_"):
                best_params_kfold.append(m.best_params_)
        if best_params_kfold:
            with open(bp_json, "w") as f:
                json.dump(best_params_kfold, f)
else:
    model_store_kfold = []
    kfold_preds = []

    for train_idx, test_idx in tqdm(kfold_splits, total=10, desc="Training 10-fold models"):
        X_train_df = df.iloc[train_idx][descriptor_cols]
        y_train_ser = df.iloc[train_idx]['solubility_g_100g_log']
        X_test_df  = df.iloc[test_idx][descriptor_cols]
        y_test_ser  = df.iloc[test_idx]['solubility_g_100g_log']
        solute_test  = df.iloc[test_idx][solute_name_col]
        solvent_test = df.iloc[test_idx]["solvent_name"]

        X_train, y_train = _to_numpy_strict(X_train_df, y_train_ser, fill_value=0)
        X_test,  y_test  = _to_numpy_strict(X_test_df,  y_test_ser, fill_value=0)


        model = RandomizedSearchCV(
            base_model, param_distributions=param_grid, n_iter=10, cv=3,
            scoring='neg_root_mean_squared_error', random_state=42,
            n_jobs=rs_n_jobs, error_score="raise",
        ) if use_random_search else base_model

        model.fit(X_train, y_train)
        model_store_kfold.append(model)
        if use_random_search and hasattr(model, "best_params_"):
            best_params_kfold.append(model.best_params_)

        fitted_model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
        y_pred = fitted_model.predict(X_test)

        temp_df = pd.DataFrame({
            'solute_name': solute_test,
            'solvent_name': solvent_test,
            'solubility_g_100g_log': y_test,   # y_test is a NumPy array; if you prefer Series: y_test_ser
            'prediction': y_pred
        })
        kfold_preds.append(temp_df)


    dump(model_store_kfold, os.path.join(base_output_path,  f"{tag}_kfold_models.joblib"))
    print("10-fold models saved.")
    # INSERT AFTER saving 10-fold models (right after: dump(model_store_kfold, ...))
    kfold_splits_path = os.path.join(base_output_path, f"{tag}_kfold_test_indices.json")
    kfold_test_indices = [test_idx.tolist() for (_, test_idx) in kfold_splits]
    with open(kfold_splits_path, "w", encoding="utf-8") as f:
        json.dump(kfold_test_indices, f, ensure_ascii=False, indent=2)
    print(f"Saved k-fold test indices: {kfold_splits_path}")
    if use_random_search and best_params_kfold:
        with open(os.path.join(base_output_path, f"{tag}_kfold_best_params.json"), "w") as f:
            json.dump(best_params_kfold, f)  # NEW


    kfold_df = pd.concat(kfold_preds, ignore_index=True)
    kfold_df.to_csv(os.path.join(prediction_dir, f"{tag}_kfold_predictions.csv"), index=False)
    print("10-fold predictions saved.")


# === Train and Save LOSO Models + Predictions ===
best_params_loso = {}  # collect best params per solute (if tuned)

loso_models_path = os.path.join(base_output_path, f"{tag}_loso_models.joblib")
loso_bp_json     = os.path.join(base_output_path, f"{tag}_loso_best_params.json")

if use_saved_models:
    # Load saved models
    model_store_loso = load(loso_models_path)

    # Ensure end up with a dict keyed by solute
    if isinstance(model_store_loso, list):
        # Try to load the saved order from JSON
        loso_order_json = os.path.join(base_output_path, f"{tag}_loso_solute_order.json")
        if os.path.exists(loso_order_json):
            with open(loso_order_json, "r", encoding="utf-8") as f:
                solute_names_order = json.load(f)
        else:
            # Fallback (less ideal): sorted unique names from current df
            solute_names_order = sorted(df[solute_name_col].unique().tolist())

        # Map list -> dict using the order
        model_store_loso = dict(zip(solute_names_order, model_store_loso))

        # Re-save as dict for all future runs
        dump(model_store_loso, loso_models_path)
        print(f"Converted legacy LOSO list -> dict and re-saved at: {loso_models_path}")

    # (unchanged) rebuild best-params JSON if needed
    if use_random_search and (not os.path.exists(loso_bp_json)):
        for sol, m in model_store_loso.items():
            if hasattr(m, "best_params_"):
                best_params_loso[sol] = m.best_params_
        if best_params_loso:
            with open(loso_bp_json, "w", encoding="utf-8") as f:
                json.dump(best_params_loso, f, ensure_ascii=False, indent=2)
            print(f"Rebuilt LOSO best params JSON: {loso_bp_json}")


else:
    model_store_loso = {}
    loso_preds = []

    # Deterministic solute order
    solute_names = sorted(df[solute_name_col].unique().tolist())
    # INSERT AFTER: save the LOSO order you actually used
    loso_order_json = os.path.join(base_output_path, f"{tag}_loso_solute_order.json")
    with open(loso_order_json, "w", encoding="utf-8") as f:
        json.dump(solute_names, f, ensure_ascii=False, indent=2)


    for solute in tqdm(solute_names, desc="Training LOSO models"):
        train_df = df[df[solute_name_col] != solute]
        test_df  = df[df[solute_name_col] == solute]

        X_train_df = train_df[descriptor_cols]
        y_train_ser = train_df['solubility_g_100g_log']
        X_test_df  = test_df[descriptor_cols]
        y_test_ser = test_df['solubility_g_100g_log']

        X_train, y_train = _to_numpy_strict(X_train_df, y_train_ser, fill_value=0)
        X_test,  y_test  = _to_numpy_strict(X_test_df,  y_test_ser, fill_value=0)


        model = RandomizedSearchCV(
            base_model, param_distributions=param_grid, n_iter=10, cv=3,
            scoring='neg_root_mean_squared_error', random_state=42,
            n_jobs=rs_n_jobs, error_score="raise",
        ) if use_random_search else base_model

        model.fit(X_train, y_train)
        model_store_loso[solute] = model
        if use_random_search and hasattr(model, "best_params_"):
            best_params_loso[solute] = model.best_params_

        fitted_model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
        y_pred = fitted_model.predict(X_test)

        temp_df = pd.DataFrame({
            'solute_name': test_df[solute_name_col].values,
            'solvent_name': test_df["solvent_name"].values,
            'solubility_g_100g_log': y_test,  # or y_test_ser.values
            'prediction': y_pred
        })
        loso_preds.append(temp_df)


    # Save models (dict) and best params JSON (if tuned)
    dump(model_store_loso, loso_models_path)
    print("LOSO models saved (dict keyed by solute).")

    if use_random_search and best_params_loso:
        with open(loso_bp_json, "w") as f:
            json.dump(best_params_loso, f)
        print("LOSO best params JSON saved.")

    # Save predictions
    loso_df = pd.concat(loso_preds, ignore_index=True)
    loso_df.to_csv(os.path.join(prediction_dir, f"{tag}_loso_predictions.csv"), index=False)
    print("LOSO predictions saved.")


if enable_y_scrambling:
    # output folder for scrambling artifacts
    scramble_dir = os.path.join(base_output_path, "scrambling")
    os.makedirs(scramble_dir, exist_ok=True)

    # safety: make sure kfold_splits exists (e.g., if you loaded models)
    if 'kfold_splits' not in locals():
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        kfold_splits = list(kf.split(df))

    # settings for this run
    SCRAMBLE_B = 200
    SPEEDY_PERMS = True  # skip tuning during permutations for speed

    # --- 10-fold permutation test ---
    obs_kf, perm_kf, p_kf = permutation_test_kfold(
        df=df,
        descriptor_cols=descriptor_cols,
        kfold_splits=kfold_splits,
        base_model=base_model,
        param_grid=param_grid,
        model_type=model_type,
        use_random_search=use_random_search,
        B=SCRAMBLE_B,
        speedy_perms=SPEEDY_PERMS,
        output_dir=scramble_dir,
        tag=tag,
    )
    print(f"[10-fold] R2={obs_kf['R2']:.3f}  p={p_kf:.4f}")

    # --- LOSO permutation test ---
    obs_loso, perm_loso, p_loso = permutation_test_loso(
        df=df,
        descriptor_cols=descriptor_cols,
        solute_name_col=solute_name_col,
        base_model=base_model,
        param_grid=param_grid,
        model_type=model_type,
        use_random_search=use_random_search,
        B=SCRAMBLE_B,
        speedy_perms=SPEEDY_PERMS,
        output_dir=scramble_dir,
        tag=tag,
    )
    print(f"[LOSO]    R2={obs_loso['R2']:.3f}  p={p_loso:.4f}")

    # --- Combined 2-panel figure (10-fold + LOSO) ---
    combined_path = os.path.join(scramble_dir, f"{tag}_r2_scrambling_combined.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    def _xlim_from(perm, obs):
        xmin = float(min(np.min(perm), obs))
        xmax = float(max(np.max(perm), obs))
        pad  = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
        return xmin - pad, xmax + pad

    # ... inside your combined figure block ...
    axes[0].hist(perm_kf, bins='auto')
    axes[0].axvline(obs_kf["R2"], linewidth=2)
    xmin, xmax = _xlim_from(perm_kf, obs_kf["R2"])
    axes[0].set_xlim(xmin, xmax)

    axes[1].hist(perm_loso, bins='auto')
    axes[1].axvline(obs_loso["R2"], linewidth=2)
    xmin, xmax = _xlim_from(perm_loso, obs_loso["R2"])
    axes[1].set_xlim(xmin, xmax)


    plt.savefig(combined_path, dpi=600); plt.close()
    print(f"Combined scrambling plot saved: {combined_path}")

    # --- 3×4 matrices + heatmaps across all runs found in the folder ---
    build_scrambling_matrices(scramble_dir)

# === Plotting & Metrics ===
from plots import (
    run_all_prediction_visualizations,
    plot_tuned_vs_cv_with_cosmo,
    export_model_metrics
)

loso_df = pd.read_csv(os.path.join(base_output_path, "predictions", f"{tag}_loso_predictions.csv"))
kfold_df = pd.read_csv(os.path.join(base_output_path, "predictions", f"{tag}_kfold_predictions.csv"))

for df_ in [loso_df, kfold_df]:
    df_['model'] = f"{model_type.upper()}-{descriptor_type} ({hybrid_str}) ({tuned_str})"
loso_df['cv_type'] = 'LOSO'
kfold_df['cv_type'] = '10Fold'


combined_df = pd.concat([loso_df, kfold_df, cosmo_df], ignore_index=True)
combined_df.to_csv(os.path.join(base_output_path, "all_combined_predictions.csv"), index=False)


run_all_prediction_visualizations(
    df=combined_df,
    output_dir=os.path.join(base_output_path, "plot_output"),
    summary_csv_path=os.path.join(base_output_path, "model_metrics_summary.csv")
)

export_model_metrics(combined_df, os.path.join(base_output_path, "model_metrics_summary.csv"))
regex = rf"{model_type}.*{descriptor_type}.*{hybrid_str}.*{tuned_str}"
plot_tuned_vs_cv_with_cosmo(
    combined_df,
    os.path.join(base_output_path, "predictions", f"{model_type.lower()}_{descriptor_type}_{hybrid_str}_tuned_LOSO_10Fold_vs_COSMO.png"),
    model_regex=regex,
    title=f"Tuned {model_type.upper()}–{descriptor_type} ({hybrid_str}): LOSO vs 10-fold vs COSMO-RS"
)

# === PCA ===
import importlib, inspect, pca_rdkit

# (optional) prove using the right file + force reload
print("[PCA] pca_rdkit file ->", pca_rdkit.__file__)
importlib.reload(pca_rdkit)

pca_out = os.path.join(base_output_path, "pca_comparison")

pca_rdkit.compare_descriptor_pca(
    df=df,
    descriptor_cols=descriptor_cols,
    descriptor_type=descriptor_type,
    solubility_col="solubility_g_100g_log",
    solute_name_col=solute_name_col,
    output_dir=pca_out,
    model_type=model_type,
    pair_3d=True,          
    show_variance=False     
)
#===SHAP Heatmap===
if use_random_search and model_type.lower() != 'rf':
    print(f"Generating SHAP heatmap for {descriptor_type.upper()} descriptors...")
    # === SHAP Heatmap ===
    if descriptor_type == 'mordred':
        from mordred_shap_heatmap import generate_mordred_shap_heatmap
        generate_mordred_shap_heatmap(
            full_df=df,
            descriptor_cols=descriptor_cols,
            model_store_kfold=model_store_kfold,
            model_store_loso=model_store_loso,
            solute_name_col=solute_name_col,
            solubility_col="solubility_g_100g_log",
            output_dir=os.path.join(base_output_path, "mordred_shap_heatmap"),
            model_type=model_type,
            kfold_test_indices=kfold_test_indices 
        )
        
    elif descriptor_type == 'moe':
        from moe_shap_heatmap import generate_moe_shap_heatmap
        generate_moe_shap_heatmap(df, descriptor_cols, model_store_kfold, model_store_loso, solute_name_col, "solubility_g_100g_log", os.path.join(base_output_path, "moe_shap_heatmap"), model_type, kfold_test_indices=kfold_test_indices)

    elif descriptor_type == 'morgan':
        from visualizer import generate_morgan_shap_heatmap

        generate_morgan_shap_heatmap(
            full_df=df,
            descriptor_cols=descriptor_cols,
            model_store_kfold=model_store_kfold,
            model_store_loso=model_store_loso,
            solute_name_col="solute_name",
            solubility_col="solubility_g_100g_log",
            output_dir=base_output_path,
            model_type="xgb",
            bit_mapping=bit_mapping,
            bitinfo=all_bitInfo,  # make sure this is passed in
            use_bit_visualization=True
        )

    elif descriptor_type == 'rdkit':
        from rdkit_shap_heatmap import generate_rdkit_shap_heatmap
        generate_rdkit_shap_heatmap(
            df,
            descriptor_cols,
            model_store_kfold,
            model_store_loso,
            solute_name_col,
            "solubility_g_100g_log",
            os.path.join(base_output_path, "rdkit_shap_heatmap"),
            model_type
        )
    if descriptor_type == 'morgan' and use_bit_visualization:
        from visualizer import generate_morgan_shap_heatmap

        generate_morgan_shap_heatmap(
            full_df=df,
            descriptor_cols=descriptor_cols,
            model_store_kfold=model_store_kfold,
            model_store_loso=model_store_loso,
            solute_name_col="solute_name",
            solubility_col="solubility_g_100g_log",
            output_dir=base_output_path,
            model_type="xgb",
            bit_mapping=bit_mapping,
            bitinfo=all_bitInfo,  # make sure this is passed in
            use_bit_visualization=True
        )
else:
    print("Skipping SHAP and visualization (no tuning). Or you used rf model.")

