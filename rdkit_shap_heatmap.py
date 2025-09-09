import os
import pandas as pd
from moe_shap_heatmap import (
    compute_global_shap_from_kfold_models,
    build_global_shap_matrix,
    compute_loso_shap_matrix,
    plot_shap_heatmap,
    annotate_descriptors,
    get_canonical_solute_order,
)



def generate_rdkit_shap_heatmap(
    full_df,
    descriptor_cols,
    model_store_kfold,
    model_store_loso,
    solute_name_col,
    solubility_col,
    output_dir,
    model_type,
    kfold_test_indices=None,
):
    os.makedirs(output_dir, exist_ok=True)
    print(" Generating SHAP heatmap for RDKit descriptors...")

    order = get_canonical_solute_order(
        full_df,
        solute_name_col=solute_name_col,
        solubility_col=solubility_col,
        agg='median',
        ascending=False,
    )

    # === 1. Global SHAP (10-fold CV) ===
    global_shap_df = compute_global_shap_from_kfold_models(
        model_list=model_store_kfold,
        df=full_df,
        descriptor_cols=descriptor_cols,
        kfold_test_indices=kfold_test_indices 
    )

    # === 2. Build heatmap matrix using 10-fold SHAP values for globally ranked top 20 descriptors ===
    top20 = global_shap_df.abs().mean().sort_values(ascending=False).head(20).index.tolist()
    top20 = [d for d in top20 if d in descriptor_cols]  # safety

    descriptor_labels = annotate_descriptors(top20)

    heatmap_matrix_kfold_global20 = build_global_shap_matrix(
        global_shap_df=global_shap_df,
        df=full_df,
        solute_name_col=solute_name_col,
        solubility_col=solubility_col,
        top_descriptors=top20
    )
    heatmap_matrix_kfold_global20 = heatmap_matrix_kfold_global20.reindex(order.intersection(heatmap_matrix_kfold_global20.index))
    heatmap_matrix_kfold_global20 = heatmap_matrix_kfold_global20.where(
        heatmap_matrix_kfold_global20.abs() >= 1e-10, 0
    )
    heatmap_matrix_kfold_global20.to_csv(os.path.join(output_dir, f"{model_type}_rdkit_shap_matrix_global20_kfold.csv"))

    # === Overlay (10-fold) ===
    activation_kfold_global20 = (
        full_df.groupby(solute_name_col)[top20]
        .first()
        .reindex(heatmap_matrix_kfold_global20.index)
        .notna()
        .astype(int)
    )


    plot_shap_heatmap(
        matrix=heatmap_matrix_kfold_global20,
        descriptor_labels=descriptor_labels,
        output_path=os.path.join(output_dir, f"{model_type}_rdkit_shap_heatmap_global20_kfold.png"),
        title=f"Top 20 RDKit Descriptors - SHAP Heatmap (10-fold, {model_type.upper()})",
        activation_matrix=activation_kfold_global20
    )

    # Save annotated top 20
    annotated_df = pd.DataFrame({
        "Descriptor": top20,
        "Mean(|SHAP|)": global_shap_df[top20].abs().mean().values,
        "Category": [descriptor_labels[d].split(" (")[1][:-1] for d in top20]
    })
    annotated_df.to_csv(os.path.join(output_dir, f"{model_type}_rdkit_top20_shap_annotated.csv"), index=False)

    # === 3. LOSO SHAP matrix ===
    heatmap_matrix_loso_global20 = compute_loso_shap_matrix(
        model_store=model_store_loso,
        df=full_df,
        descriptor_cols=descriptor_cols,
        solute_name_col=solute_name_col,
        solubility_col=solubility_col,
        top_descriptors=top20,
        order=order
    )
    heatmap_matrix_loso_global20.to_csv(os.path.join(output_dir, f"{model_type}_rdkit_shap_matrix_global20_loso.csv"))

    # === Overlay (LOSO) ===
    activation_loso_global20 = (
        full_df.groupby(solute_name_col)[top20]
        .first()   # use .max(numeric_only=True) or .any() if prefer "any occurrence"
        .reindex(heatmap_matrix_loso_global20.index)
        .notna()
        .astype(int)
    )

    plot_shap_heatmap(
        matrix=heatmap_matrix_loso_global20,
        descriptor_labels=descriptor_labels,
        output_path=os.path.join(output_dir, f"{model_type}_rdkit_shap_heatmap_global20_loso.png"),
        title=f"Top 20 RDKit Descriptors - SHAP Heatmap (LOSO, {model_type.upper()})",
        activation_matrix=activation_loso_global20
    )

    print(" RDKit SHAP heatmaps and matrix saved.")
