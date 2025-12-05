import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from visualizer import (
    compute_shap_matrix,
    generate_bit_heatmap_matrix,
    create_shap_heatmap_with_activation,
    load_bit_mapping,
    visualize_top_bits,
    summarize_bit_frequency,
)

def compute_loso_shap_and_visualize(
    model_store,
    df,
    descriptor_cols,
    solute_name_col,
    solubility_col,
    smiles_col,
    base_output_path,
    model_type,
    bit_mapping_path
):
    print("Computing SHAP values from LOSO-CV models...")

    solute_names = df[solute_name_col].unique()
    all_loso_shap = []

    for solute, model in zip(solute_names, model_store):
        test_df = df[df[solute_name_col] == solute]
        X_test = test_df[descriptor_cols].astype(float)

        fitted_model = model.best_estimator_ if isinstance(model, RandomizedSearchCV) else model
        shap_values = compute_shap_matrix(fitted_model, X_test, descriptor_cols)
        shap_values['solute'] = solute
        all_loso_shap.append(shap_values)

    loso_shap_df = pd.concat(all_loso_shap, axis=0)

    # === Build heatmap ===
    top20_bits = loso_shap_df.drop(columns='solute').abs().mean().sort_values(ascending=False).head(20).index.tolist()

    shap_matrix, activation_matrix = generate_bit_heatmap_matrix(
        df, loso_shap_df.drop(columns='solute'), top20_bits,
        solute_name_col=solute_name_col,
        solubility_col=solubility_col
    )

    output_path = os.path.join(base_output_path, f"{model_type}_morgan_loso_shap_heatmap_overlay.png")
    create_shap_heatmap_with_activation(
        shap_df=shap_matrix,
        activation_matrix=activation_matrix,
        title="Top 20 Morgan Bits SHAP Heatmap (LOSO-CV)",
        output_path=output_path
    )

    bit_mapping = load_bit_mapping(bit_mapping_path)
    visualize_top_bits(
        top_bits=top20_bits,
        df=df,
        smiles_col=smiles_col,
        bit_mapping=bit_mapping,
        output_dir=os.path.join(base_output_path, "bit_visuals_loso")
    )

    summarize_bit_frequency(df, top20_bits, base_output_path)
    print("LOSO SHAP heatmap and bit visualizations generated.")
