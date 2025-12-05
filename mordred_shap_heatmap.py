import os
import pandas as pd
from moe_shap_heatmap import (
    compute_global_shap_from_kfold_models,
    build_global_shap_matrix,
    compute_loso_shap_matrix,
    plot_shap_heatmap,
    annotate_descriptors
)

def generate_mordred_shap_heatmap(
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
    print("Generating SHAP heatmap for Mordred descriptors...")
    # 0) Shared row order (highest -> lowest mean solubility)
    def _default_order(df, target_col="solubility_g_100g_log", id_col="solute_name"):
        return (
            df.groupby(id_col, as_index=False)[target_col]
            .mean()
            .sort_values(target_col, ascending=False)[id_col]
            .tolist()
        )

    order = _default_order(full_df, target_col=solubility_col, id_col=solute_name_col)

    # === 1. Global SHAP (10-fold CV) ===
    global_shap_df = compute_global_shap_from_kfold_models(
        model_list=model_store_kfold,
        df=full_df,
        descriptor_cols=descriptor_cols,
        kfold_test_indices=kfold_test_indices 
    )

    # === 2. Identify top 20 descriptors globally ===
    top20 = global_shap_df.abs().mean().sort_values(ascending=False).head(20).index.tolist()
    top20 = [d for d in top20 if d in descriptor_cols]
    descriptor_labels = annotate_descriptors(top20)

    # === 3. Heatmap matrix (10-fold CV) ===
    heatmap_matrix_kfold = build_global_shap_matrix(
        global_shap_df=global_shap_df,
        df=full_df,
        solute_name_col=solute_name_col,
        solubility_col=solubility_col,
        top_descriptors=top20
    )
    # threshold tiny values -> 0 (no applymap)
    heatmap_matrix_kfold = heatmap_matrix_kfold.reindex(order)   
    heatmap_matrix_kfold = heatmap_matrix_kfold.mask(heatmap_matrix_kfold.abs() < 1e-10, 0)

    activation_kfold = full_df.groupby(solute_name_col)[top20].first().reindex(heatmap_matrix_kfold.index)  .notna().astype(int)

    plot_shap_heatmap(
        matrix=heatmap_matrix_kfold,
        descriptor_labels=descriptor_labels,
        output_path=os.path.join(output_dir, f"{model_type}_mordred_shap_heatmap_global20_kfold.png"),
        title=f"Top 20 Mordred Descriptors - SHAP Heatmap (10-fold, {model_type.upper()})",
        activation_matrix=activation_kfold
    )

    # Save annotation table
    annotated_df = pd.DataFrame({
        "Descriptor": top20,
        "Mean(|SHAP|)": global_shap_df[top20].abs().mean().values,
        "Category": [descriptor_labels[d].split(" (")[1][:-1] for d in top20]
    })
    annotated_df.to_csv(os.path.join(output_dir, f"{model_type}_mordred_top20_shap_annotated.csv"), index=False)

    def _default_order(df, target_col="solubility_g_100g_log", id_col="solute_name"):
        order = (
            df.groupby(id_col, as_index=False)[target_col]
            .mean()
            .sort_values(target_col)[id_col]
            .tolist()
        )
        return order
    # === 4. LOSO SHAP matrix ===
    

    heatmap_matrix_loso = compute_loso_shap_matrix(
        model_store=model_store_loso,
        df=full_df,
        descriptor_cols=descriptor_cols,
        solute_name_col=solute_name_col,
        solubility_col=solubility_col,
        top_descriptors=top20
    )
    heatmap_matrix_loso = heatmap_matrix_loso.reindex(order)   
    heatmap_matrix_loso.to_csv(os.path.join(output_dir, f"{model_type}_mordred_shap_matrix_global20_loso.csv"))

    activation_loso = (
        full_df.groupby(solute_name_col)[top20]
        .first()
        .reindex(heatmap_matrix_loso.index)
        .notna()
        .astype(int)
    )


    plot_shap_heatmap(
        matrix=heatmap_matrix_loso,
        descriptor_labels=descriptor_labels,
        output_path=os.path.join(output_dir, f"{model_type}_mordred_shap_heatmap_global20_loso.png"),
        title=f"Top 20 Mordred Descriptors - SHAP Heatmap (LOSO, {model_type.upper()})",
        activation_matrix=activation_loso
    )

    print("Mordred SHAP heatmaps and matrix saved.")
