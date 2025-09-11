import os
import numpy as np
import pandas as pd
import shap
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import random
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

random.seed(42)
np.random.seed(42)

shap_cmap = sns.diverging_palette(220, 20, as_cmap=True)



# === Descriptor Annotation ===
def annotate_descriptors(descriptor_names):
    category_map = {

        'cosmo': 'COSMO-RS',
        # Hydrophobicity
        'SlogP': 'Hydrophobicity',
        'ALogP': 'Hydrophobicity',
        'LogP': 'Hydrophobicity',

        # Surface Area
        'VSA': 'Surface Area',
        'vsa': 'Surface Area',
        'LabuteASA': 'Surface Area',

        # Polarity
        'PEOE': 'Polarity',
        'TPSA': 'Polarity',
        'QED': 'Polarity',

        # Electronic
        'EState': 'Electronic',
        'PEOE_VSA': 'Electronic',

        # H-Bonding
        'vsa_don': 'H-Bonding',
        'vsa_acc': 'H-Bonding',
        'nHBDon': 'H-Bonding',
        'nHBAcc': 'H-Bonding',

        # Refractivity
        'SMR': 'Refractivity',
        'AMR': 'Refractivity',

        # Topological
        'Zagreb': 'Topological',
        'Chi': 'Topological',
        'Kappa': 'Topological',

        # Connectivity/Shape
        'vsurf': 'Shape',
        'VolSurf': 'Shape',

        # Size
        'MW': 'Size',
        'Weight': 'Size'
    }


    def match_category(desc):
        for key, cat in category_map.items():
            if key in desc:
                return cat
        return "Uncategorized"

    return {desc: f"{desc} ({match_category(desc)})" for desc in descriptor_names}

# === Modify heatmap label coloring ===
def color_code_tick_labels(ax):
    for tick_label in ax.get_xticklabels():
        label_text = tick_label.get_text()
        if label_text.startswith("solute_"):
            tick_label.set_color("#1f77b4")  # blue
        elif label_text.startswith("solvent_"):
            tick_label.set_color("#2ca02c")  # green

def get_canonical_solute_order(df, solute_name_col, solubility_col, agg="mean", ascending=False):
    g = df[[solute_name_col, solubility_col]].groupby(solute_name_col)[solubility_col]
    s = getattr(g, agg)().dropna()
    return s.sort_values(ascending=ascending).index

def get_shap_explainer(model, X, n_background=10, X_background=None):
    """
    Returns a SHAP explainer for the given fitted model and feature frame X.
    - If the final estimator is tree-based (XGB, RandomForest), uses TreeExplainer
      with an interventional baseline aligned to train-time columns.
    - Otherwise, uses KernelExplainer with a train-based KMeans background
      (deterministic, thread-limited to avoid MKL warning on Windows).

    Parameters
    ----------
    model : fitted estimator or Pipeline
    X : pd.DataFrame
        Points plan to explain (e.g., test fold).
    n_background : int
        Number of KMeans clusters (caps background size for KernelExplainer).
    X_background : pd.DataFrame or None
        Background source (e.g., training fold). If None, falls back to X.
    """
    # --- unwrap CV wrappers and pipelines
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    pipe = model if isinstance(model, Pipeline) else None
    final_est = pipe.steps[-1][1] if pipe is not None else model
    model_name = type(final_est).__name__.lower()

    # --- utility: align a frame's columns to what the model/scaler saw at train time
    def _align_to_trained_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[:, df.notna().any(axis=0)].copy()
        # If a scaler exists with feature_names_in_, prefer that (pipelines with SVM etc.)
        if pipe is not None and "scaler" in getattr(pipe, "named_steps", {}):
            scaler = pipe.named_steps["scaler"]
            if hasattr(scaler, "feature_names_in_"):
                return df.reindex(columns=list(scaler.feature_names_in_), fill_value=0).fillna(0)
        # Else try estimator feature_names_in_ (supported by many sklearn models)
        if hasattr(final_est, "feature_names_in_"):
            return df.reindex(columns=list(final_est.feature_names_in_), fill_value=0).fillna(0)
        # Fallback: numeric only
        return df.select_dtypes(include=[np.number]).fillna(0)

    # --- pretty route log
    is_tree = ("xgb" in model_name) or ("randomforest" in model_name) or hasattr(final_est, "get_booster")
    print(f" SHAP route: final_est={type(final_est).__name__} -> "
          f"{'TreeExplainer' if is_tree else 'KernelExplainer'}")

    # =========================
    # Route A: TreeExplainer
    # =========================
    if is_tree:
        # background from TRAIN if provided, else X (both aligned)
        bg_src = _align_to_trained_columns(X_background if X_background is not None else X)
        # shrink background for speed
        if len(bg_src) > 100:
            bg_src = bg_src.sample(n=100, random_state=0)
        bg_src = bg_src.astype("float32", copy=False)

        # Interventional TreeExplainer (good for XGB/RF)
        return shap.TreeExplainer(
            final_est,
            data=bg_src,
            feature_perturbation="interventional",
            model_output="raw"  # explains margin for XGB; keep consistent when checking additivity
        )

    # =========================
    # Route B: KernelExplainer
    # =========================
    # Points to explain (X) and background source (prefer TRAIN)
    X_clean = _align_to_trained_columns(X)
    if X_clean.shape[1] == 0:
        raise ValueError("All descriptor columns are NaN — SHAP cannot proceed.")
    bg_src = _align_to_trained_columns(X_background) if X_background is not None else X_clean
    bg_src = bg_src.astype("float32", copy=False)

    # pick deterministic KMeans background, avoid MKL Windows warning
    n_clusters = max(1, min(n_background, bg_src.shape[0]))
    print(f" SHAP using KMeans background with {n_clusters} clusters")
    with threadpool_limits(limits=1):
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10, algorithm="lloyd")
        centers = km.fit(bg_src.values).cluster_centers_

    # robust predict wrapper (accepts ndarray or DataFrame, aligns columns)
    def model_predict(X_input):
        if isinstance(X_input, np.ndarray):
            X_input = pd.DataFrame(X_input, columns=X_clean.columns)
        X_input = _align_to_trained_columns(X_input)
        return model.predict(X_input)

    return shap.KernelExplainer(model_predict, centers)



# === Step 1: SHAP from 10-fold for global ranking ===
def compute_global_shap_from_kfold_models(model_list, df, descriptor_cols, kfold_test_indices=None):

    all_shap = []

    if kfold_test_indices is None:
        # fallback to internal split (kept for backward-compat)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        split_iter = [test_idx for _, test_idx in kf.split(df)]
    else:
        split_iter = [np.array(ti, dtype=int) for ti in kfold_test_indices]

    for model, test_idx in tqdm(zip(model_list, split_iter),
                            total=len(model_list), desc="SHAP (10-fold)"):

        # test / train split for this fold
        test_df  = df.iloc[test_idx]
        train_idx = np.setdiff1d(np.arange(len(df)), test_idx)
        train_df = df.iloc[train_idx]

        # features (align types once)
        X_test  = test_df.loc[:, descriptor_cols].astype(float)
        X_train = train_df.loc[:, descriptor_cols].astype(float)

        # unwrap and build explainer (pass TRAIN as background)
        fitted_model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
        explainer = get_shap_explainer(
            fitted_model,
            X=X_test,
            n_background=10,
            X_background=X_train              # train-based baseline for SVM/Kernel
        )

        # version-safe SHAP value computation
        Xf = X_test.astype(np.float32, copy=False)
        if isinstance(explainer, shap.KernelExplainer):
            p = Xf.shape[1]
            nsamples = min(max(2*p + 1, 100), 300)
            values = explainer.shap_values(Xf, nsamples=nsamples, l1_reg="num_features(10)")
        elif hasattr(explainer, "shap_values"):          # TreeExplainer on many versions
            values = explainer.shap_values(Xf)
        else:                                             # newer unified API
            values = explainer(Xf).values

        shap_df = pd.DataFrame(values, columns=descriptor_cols, index=test_df.index)
        all_shap.append(shap_df)


    return pd.concat(all_shap, axis=0)


def build_global_shap_matrix(global_shap_df, df, solute_name_col, solubility_col, top_descriptors):
    matrix = []
    solutes = []

    for solute in df[solute_name_col].unique():
        idx = df[df[solute_name_col] == solute].index
        solute_shap = global_shap_df.loc[idx]

        if solute_shap.empty:
            continue

        row = solute_shap[top_descriptors].mean().reindex(top_descriptors, fill_value=0)
        matrix.append(row)
        solutes.append(solute)

    heatmap_df = pd.DataFrame(matrix, index=solutes, columns=top_descriptors)
    solute_sol = df.groupby(solute_name_col)[solubility_col].mean()
    heatmap_df['solubility'] = [solute_sol.get(s, np.nan) for s in solutes]
    heatmap_df = heatmap_df.sort_values('solubility', ascending=False).drop(columns='solubility')
    # Drop any non-descriptor columns (safety check)
    heatmap_df = heatmap_df[top_descriptors]

    return heatmap_df


# === Step 2: SHAP matrix from LOSO ===
def compute_loso_shap_matrix(model_store, df, descriptor_cols, solute_name_col, solubility_col, top_descriptors):
    all_rows = []
    # Build (model, solute) pairs robustly for dict OR list
    solute_order = df.groupby(solute_name_col)[solubility_col].mean().sort_values(ascending=False).index

    if isinstance(model_store, dict):
        pairs = []
        missing = []
        for s in solute_order:
            m = model_store.get(s, None)
            if m is None:
                missing.append(s)
            else:
                pairs.append((m, s))
        if missing:
            print(f"{len(missing)} solutes missing in model_store (skipping): {missing[:5]}...")
    else:
        # legacy list: zip with order (will be correct if converted at load time)
        pairs = list(zip(model_store, solute_order))
    print("DEBUG: First few top_descriptors:", top_descriptors[:5])

    for model, solute in tqdm(pairs, total=len(pairs), desc="SHAP (LOSO)"):

        test_df  = df[df[solute_name_col] == solute]
        train_df = df[df[solute_name_col] != solute]

        X_test  = test_df[descriptor_cols].astype(float)
        X_train = train_df[descriptor_cols].astype(float)

        fitted_model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
        explainer = get_shap_explainer(
            fitted_model,
            X=X_test,
            n_background=10,
            X_background=X_train              # train-based baseline for SVM/Kernel
        )

        Xf = X_test.astype(np.float32, copy=False)
        if isinstance(explainer, shap.KernelExplainer):
            p = Xf.shape[1]
            nsamples = min(max(2*p + 1, 100), 300)
            values = explainer.shap_values(Xf, nsamples=nsamples, l1_reg="num_features(10)")
        elif hasattr(explainer, "shap_values"):
            values = explainer.shap_values(Xf)
        else:
            values = explainer(Xf).values



        # === Check shape match
        assert values.shape[1] == len(descriptor_cols), f"Mismatch: {values.shape[1]} vs {len(descriptor_cols)}"

        print(f"SHAP mean for solute {solute}: {np.mean(np.abs(values)):.2e}")
        print("Full SHAP value range:", values.min(), "to", values.max())


        # === Construct shap_df and strip non-numeric
        shap_df = pd.DataFrame(values, columns=descriptor_cols, index=test_df[solute_name_col].values)
        shap_df = shap_df.select_dtypes(include=[np.number])  #  Just to be safe



        # === Confirm descriptor columns exist
        print("DEBUG: SHAP columns (first 5):", shap_df.columns[:5].tolist())
        print("shap_df shape:", shap_df.shape)
        print("test_df shape:", test_df.shape)
        print("solute values:", test_df[solute_name_col].unique())

        # === Filter to top descriptors
        common = [d for d in top_descriptors if d in shap_df.columns]
        missing = [d for d in top_descriptors if d not in shap_df.columns]
        print(f"{solute} – Missing {len(missing)} of {len(top_descriptors)} descriptors: {missing[:5]}")

        if not common:
            print(f"Skipping {solute}: No top descriptors found in SHAP output.")
            continue

        row = pd.DataFrame([shap_df[common].mean()], index=[solute])


        # ===  Safety check
        if 'solute' in row.columns:
            row = row.drop(columns='solute')

        row.index = [solute]
        row = row.reindex(columns=top_descriptors, fill_value=0)
        print("Top 5 SHAP values in row:", row.iloc[0].abs().sort_values(ascending=False).head(5))
        row_sum = row.abs().sum(axis=1)
        print(f"Total SHAP contribution for {solute}: {row_sum.values[0]:.4e}")


        # ===  Deep check on output row
        print(f"\n====== {solute} Summary ======")
        print("SHAP value range:", row.min().min(), "to", row.max().max())
        if row.isnull().all(axis=None):
            print(f" All values NaN for solute {solute} – investigate!")
        elif (row.abs() < 1e-20).all(axis=None):
            print(f" All values effectively zero for solute {solute} – investigate!")
        else:
            print(f" Row for {solute} looks OK.")
        print("Row preview:\n", row.iloc[:, :5])
        print("================================\n")

        all_rows.append(row)

    # === Combine and clean up ===
    heatmap_df = pd.concat(all_rows)
    valid_solutes = [s for s in solute_order if s in heatmap_df.index]
    heatmap_df = heatmap_df.loc[valid_solutes]
    heatmap_df = heatmap_df[top_descriptors]  # enforce order

    print(" Final SHAP matrix shape:", heatmap_df.shape)
    print("Non-zero values:", (heatmap_df.abs() > 1e-20).sum().sum())

    return heatmap_df

# === Step 3: Plot heatmap ===
def plot_shap_heatmap(matrix, descriptor_labels, output_path, title, activation_matrix=None):
    print("\n Raw SHAP matrix before thresholding:")

    # Ensure only the top descriptors are included
    matrix = matrix.loc[:, list(descriptor_labels.keys())]


    print(matrix.head(5))  # Print first 5 rows

    # Clean small values
    matrix_renamed = matrix.copy()
    for col in matrix_renamed.columns:
        matrix_renamed[col] = matrix_renamed[col].map(
            lambda x: 0 if pd.notna(x) and abs(x) < 1e-30 else x
        )

    # Format for annotation
    def format_shap_value(val):
        if pd.isna(val) or abs(val) < 1e-30:
            return "0"
        exponent = int(np.floor(np.log10(abs(val))))
        sign = "+" if val > 0 else "-"
        return f"{sign}10^{exponent}"


    plt.figure(figsize=(20, 16), dpi=300)
    abs_max = np.nanpercentile(np.abs(matrix_renamed.values), 95)
    vmin, vmax = -abs_max, abs_max
    ax = sns.heatmap(
        matrix_renamed,
        cmap=shap_cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        annot=matrix_renamed.map(lambda x: format_shap_value(x)).values,
        fmt="",
        annot_kws={"size": 4},
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"label": "Signed SHAP Value"}
    )
    color_code_tick_labels(ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Top Descriptors (Category)")
    ax.set_ylabel("Solutes\n(decreasing solubility ↓)", fontsize=12, rotation=0, labelpad=50)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # PNG
    plt.savefig(output_path.replace(".png", ".tiff"), dpi=600, format="tiff", bbox_inches="tight")  # TIFF
    plt.close()

# === Combined SHAP heatmap function ===
def generate_moe_shap_heatmap(
    full_df,
    descriptor_cols,
    model_store_kfold,
    model_store_loso,
    solute_name_col,
    solubility_col,
    output_dir,
    model_type,
    kfold_test_indices=None
):
    os.makedirs(output_dir, exist_ok=True)
    print(" Generating SHAP heatmap for MOE descriptors...")

    # === 1. Global SHAP (10-fold CV) ===
    global_shap_df = compute_global_shap_from_kfold_models(
        model_list=model_store_kfold,
        df=full_df,
        descriptor_cols=descriptor_cols,
        kfold_test_indices=kfold_test_indices 
    )

    # === 2. Build heatmap matrix using 10-fold SHAP values for globally ranked top 20 descriptors ===
    # ===  Identify top 20 globally important descriptors ===
    top20 = global_shap_df.abs().mean().sort_values(ascending=False).head(20).index.tolist()
    # Remove non-descriptor columns from top20
    top20 = [d for d in top20 if d in descriptor_cols]

    descriptor_labels = annotate_descriptors(top20)

    heatmap_matrix_kfold_global20 = build_global_shap_matrix(
        global_shap_df=global_shap_df,
        df=full_df,
        solute_name_col=solute_name_col,
        solubility_col=solubility_col,
        top_descriptors=top20
    )

    heatmap_matrix_kfold_global20 = heatmap_matrix_kfold_global20.mask(
        heatmap_matrix_kfold_global20.abs() < 1e-10, 0
    )

    heatmap_matrix_kfold_global20.to_csv(os.path.join(output_dir, f"{model_type}_moe_shap_matrix_global20_kfold.csv"))

    # === Activation overlay for global descriptors (10-fold SHAP) ===
    activation_kfold_global20 = full_df.groupby(solute_name_col)[top20].first().loc[heatmap_matrix_kfold_global20.index].notna().astype(int)

    # === Plot: Global Top 20 Heatmaps (10-fold SHAP) ===
    plot_shap_heatmap(
        matrix=heatmap_matrix_kfold_global20,
        descriptor_labels=descriptor_labels,
        output_path=os.path.join(output_dir, f"{model_type}_moe_shap_heatmap_global20_kfold.png"),
        title=f"Top 20 MOE Descriptors - SHAP Heatmap (10-fold, {model_type.upper()})"
    )

    # Save global descriptor annotations
    annotated_df = pd.DataFrame({
        "Descriptor": top20,
        "Mean(|SHAP|)": global_shap_df[top20].abs().mean().values,
        "Category": [descriptor_labels[d].split(" (")[1][:-1] for d in top20]
    })
    annotated_df.to_csv(os.path.join(output_dir, f"{model_type}_moe_top20_shap_annotated.csv"), index=False)

    # === 3. LOSO SHAP matrix (global top 20 descriptors) ===
    heatmap_matrix_loso_global20 = compute_loso_shap_matrix(
        model_store=model_store_loso,
        df=full_df,
        descriptor_cols=descriptor_cols,
        solute_name_col=solute_name_col,
        solubility_col=solubility_col,
        top_descriptors=top20
    )
    #heatmap_matrix_loso_global20 = heatmap_matrix_loso_global20.applymap(lambda x: 0 if abs(x) < 1e-10 else x)
    heatmap_matrix_loso_global20.to_csv(os.path.join(output_dir, f"{model_type}_moe_shap_matrix_global20_loso.csv"))


    # === 4. Activation overlay matrix (global) ===
    activation_loso_global20 = full_df.set_index(solute_name_col).loc[heatmap_matrix_loso_global20.index, top20].notna().astype(int)

    plot_shap_heatmap(
        matrix=heatmap_matrix_loso_global20,
        descriptor_labels=descriptor_labels,
        output_path=os.path.join(output_dir, f"{model_type}_moe_shap_heatmap_global20_loso.png"),
        title=f"Top 20 MOE Descriptors - SHAP Heatmap (LOSO, {model_type.upper()})"
    )

    print(" Global SHAP heatmaps and matrix saved.")

