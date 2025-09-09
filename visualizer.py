import os
import pandas as pd
import seaborn as sns
from bit_analysis import (
    summarize_bit_frequency,
    visualize_top_bits_with_bitinfo
)
from moe_shap_heatmap import (
    compute_global_shap_from_kfold_models,
    compute_loso_shap_matrix,
    plot_shap_heatmap,
    get_canonical_solute_order
)



shap_cmap = sns.diverging_palette(220, 20, as_cmap=True)

def generate_morgan_shap_heatmap(
    full_df,
    descriptor_cols,
    model_store_kfold,
    model_store_loso,
    solute_name_col,
    solubility_col,
    output_dir,
    model_type,
    bit_mapping,
    bitinfo,  # <== required!
    use_bit_visualization=True
):
    order = get_canonical_solute_order(
        full_df, solute_name_col=solute_name_col, solubility_col=solubility_col,
        agg='median', ascending=False
    )

    global_shap_df = compute_global_shap_from_kfold_models(
            model_list=model_store_kfold,
            df=full_df,
            descriptor_cols=descriptor_cols
        )
    if global_shap_df is None or global_shap_df.empty:
        print("⚠️ global_shap_df is empty — skipping Morgan SHAP heatmaps.")
        return

    for role in ["solute", "solvent"]:
        role_output_dir = os.path.join(output_dir, f"{model_type}_{role}_bit_analysis")
        os.makedirs(role_output_dir, exist_ok=True)

        # Filter descriptor cols specific to this role
        role_bit_cols = [c for c in descriptor_cols if c.startswith(f"{role}_FP_") and c.split("_")[-1].isdigit()]
        if not role_bit_cols:
            print(f"⚠️ No {role} fingerprint columns found — skipping {role}.")
            continue

        # === Select top 20 bits by mean absolute SHAP
        top20 = (
            global_shap_df[role_bit_cols]
            .abs()
            .mean()
            .sort_values(ascending=False)
            .head(20)
            .index.tolist()
        )

        # Save raw top bit names before cleaning
        with open(os.path.join(role_output_dir, f"top_shap_bits_raw_{role}.txt"), "w") as f:
            f.write("\n".join(top20))

        global_shap_df[top20].to_csv(os.path.join(role_output_dir, f"{model_type}_{role}_shap_top20_global.csv"))

        # === Heatmap (10-fold)
        matrix_kfold = global_shap_df[top20].groupby(full_df[solute_name_col]).mean()
        matrix_kfold = matrix_kfold.reindex(order.intersection(matrix_kfold.index))

        activation_kfold = (
            full_df.groupby(solute_name_col)[top20].first().astype(bool).astype(int)
            .reindex(matrix_kfold.index)
        )

        matrix_kfold = matrix_kfold.where(matrix_kfold.abs() >= 1e-30, 0.0)
        matrix_kfold.to_csv(os.path.join(role_output_dir, f"{model_type}_{role}_shap_matrix_global20_kfold.csv"))

        descriptor_labels = {col: col.split("_FP_")[-1] for col in top20}

        plot_shap_heatmap(
            matrix=matrix_kfold,
            descriptor_labels=descriptor_labels,
            output_path=os.path.join(role_output_dir, f"{model_type}_{role}_bit_shap_heatmap_global20_kfold.png"),
            title=f"Top 20 {role.capitalize()} Morgan Bits – SHAP Heatmap (10-fold, {model_type.upper()})",
            activation_matrix=activation_kfold
        )

        # === Heatmap (LOSO)
        matrix_loso = compute_loso_shap_matrix(
            model_store=model_store_loso,
            df=full_df,
            descriptor_cols=descriptor_cols,
            solute_name_col=solute_name_col,
            solubility_col=solubility_col,
            top_descriptors=top20,
        )
        if matrix_loso is None or matrix_loso.empty:
            print("⚠️ LOSO SHAP matrix is empty — skipping LOSO heatmap for this role.")
            # still write an empty CSV so downstream scripts don’t break
            empty_path = os.path.join(role_output_dir, f"{model_type}_{role}_shap_matrix_global20_loso.csv")
            pd.DataFrame(columns=top20, index=[]).to_csv(empty_path, index=True)
            # and skip the plotting part for LOSO
            continue
        if order is not None:
            # keep only those present in the matrix; avoids KeyError
            order_filtered = [s for s in order if s in matrix_loso.index]
            matrix_loso = matrix_loso.reindex(order_filtered)

        matrix_loso = matrix_loso.reindex(order.intersection(matrix_loso.index))

        activation_loso = (
            full_df.set_index(solute_name_col).loc[matrix_loso.index, top20]
            .astype(bool).astype(int)
        )
        matrix_loso.to_csv(os.path.join(role_output_dir, f"{model_type}_{role}_shap_matrix_global20_loso.csv"))

        plot_shap_heatmap(
            matrix=matrix_loso,
            descriptor_labels=descriptor_labels,
            output_path=os.path.join(role_output_dir, f"{model_type}_{role}_bit_shap_heatmap_global20_loso.png"),
            title=f"Top 20 {role.capitalize()} Morgan Bits – SHAP Heatmap (LOSO, {model_type.upper()})",
            activation_matrix=activation_loso
        )

        top_bits_cleaned = [bit_mapping[col] for col in top20 if col in bit_mapping]
        summarize_bit_frequency(full_df, top_bits_cleaned, role_output_dir, role=role)

        print(f" Extracted bit indices (cleaned): {top_bits_cleaned}")


        if use_bit_visualization:
            pd.Series([b for _, b in top_bits_cleaned], name="global_bit_index").to_csv(
                os.path.join(role_output_dir, f"{model_type}_{role}_top20_global_bit_ids.csv"), index=False
            )


            print(f"\n Top 20 {role} bit columns: {top20}")
            print(f" Extracted bit indices (cleaned): {top_bits_cleaned}")
            print(f" Bit mapping preview: {list(bit_mapping.items())[:5]}")

            visualize_top_bits_with_bitinfo(
                top_bits=top_bits_cleaned,
                df=full_df,
                smiles_col=f"{role}_smiles",
                bit_mapping=bit_mapping,
                output_dir=role_output_dir,
                role=role,
                bitinfo_data=bitinfo[role], # now index-based!
                sub_img_size=(600, 600) 
            )



    print(" Morgan SHAP heatmaps, summaries, and bit images saved.")
