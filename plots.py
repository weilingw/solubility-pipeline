# prediction_plot_utils.py

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from matplotlib.lines import Line2D

# after plotting scatter points
handles = [Line2D([0], [0], marker='o', color='w', label=f"{cv_type} models",
                  markerfacecolor=cv_colors[cv_type], markersize=8)
           for cv_type in df['cv_type'].unique()]
plt.legend(handles=handles, title="CV Type", loc="best")


# === Color Scheme by Model Type ===
cv_colors = {
    "COSMO": "#5B9BD5",   # Blue
    "LOSO": "#70AD47",    # Green
    "10Fold": "#A64D79"   # Purple/Rose
}

def normalize_model_label(model_label):
    parts = model_label.split("-")
    if parts:
        parts[0] = parts[0].lower()
    return "-".join(parts)

def get_model_color_map(model_names):
    color_map = {}
    default_colors = {
        'xgb': '#FF6F61',        # Coral Red
        'rf': '#70AD47',         # Green
        'svm': '#6A5ACD',        # Slate Blue
        'cosmo-rs': '#5B9BD5'    # Blue
    }
    
    for model in model_names:
        model_key = model.lower()
        if "cosmo" in model_key:
            color_map[model] = default_colors['cosmo-rs']
        elif model_key.startswith("xgb"):
            color_map[model] = default_colors['xgb']
        elif model_key.startswith("rf"):
            color_map[model] = default_colors['rf']
        elif model_key.startswith("svm"):
            color_map[model] = default_colors['svm']
        else:
            color_map[model] = 'grey'
    return color_map

# === COSMO baseline loading (optional) ===
def load_cosmo_baseline(cosmo_path):
    cosmo_df = pd.read_csv(cosmo_path)
    cosmo_df['model'] = "COSMO-RS (Baseline)"
    cosmo_df['cv_type'] = "COSMO"
    cosmo_df = cosmo_df.rename(columns={"cosmo_prediction": "prediction"})
    return cosmo_df

# === Evaluation Metrics ===
def export_model_metrics(df, output_path):
    metrics = []
    grouped = df.groupby(['model', 'cv_type'])
    for (model, cv_type), group in grouped:
        true = group['solubility_g_100g_log']
        pred = group['prediction']
        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        metrics.append([normalize_model_label(model), cv_type, round(r2, 4), round(rmse, 4), round(mae, 4)])
    summary_df = pd.DataFrame(metrics, columns=['Model', 'CV_Type', 'R2', 'RMSE', 'MAE'])
    summary_df.to_csv(output_path, index=False)
    print(f" Model evaluation metrics saved to: {output_path}")

# === Combined Parity Plots by CV Type ===
def plot_combined_by_cv_type(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for cv_type, group in df.groupby("cv_type"):
        plt.figure(figsize=(6, 6), dpi=300)
        for model in group['model'].unique():
            model_df = group[group['model'] == model]
            plt.scatter(model_df['solubility_g_100g_log'], model_df['prediction'], label=normalize_model_label(model), alpha=0.7, color=cv_colors.get(cv_type, "gray"))
        min_val = min(group['solubility_g_100g_log'].min(), group['prediction'].min()) - 0.5
        max_val = max(group['solubility_g_100g_log'].max(), group['prediction'].max()) + 0.5
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
        plt.plot([min_val, max_val], [min_val + 1, max_val + 1], 'r--', lw=0.75)
        plt.plot([min_val, max_val], [min_val - 1, max_val - 1], 'r--', lw=0.75)
        plt.title(f'Combined Parity Plot - {cv_type}')
        plt.xlabel('True Solubility (log g/100g)')
        plt.ylabel('Predicted Solubility')
        plt.legend(title="Model", loc="best")
        plt.suptitle(f"Cross-validation type: {cv_type}", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{cv_type}_combined_parity.png"))
        plt.close()

# === Individual Parity Plots ===
def plot_individual_model_results(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    color_map = get_model_color_map(df['model'].unique())
    for model_name, group in df.groupby("model"):
        label = normalize_model_label(model_name)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.scatter(group['solubility_g_100g_log'], group['prediction'], edgecolor='k', alpha=0.7, color=color_map[model_name])
        min_val = min(group['solubility_g_100g_log'].min(), group['prediction'].min()) - 0.5
        max_val = max(group['solubility_g_100g_log'].max(), group['prediction'].max()) + 0.5
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
        ax.plot([min_val, max_val], [min_val + 1, max_val + 1], 'r--', lw=0.75)
        ax.plot([min_val, max_val], [min_val - 1, max_val - 1], 'r--', lw=0.75)
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_title(f'Parity Plot - {label}')
        ax.set_xlabel('True Solubility (log g/100g)')
        ax.set_ylabel('Predicted Solubility')
        ax.legend()
        ax.grid(True)
        filename = re.sub(r'[^A-Za-z0-9]+', '_', label) + "_parity.png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

# === Cumulative Error Plot ===
def plot_cumulative_error(df, output_dir):
    plt.figure(figsize=(6, 5), dpi=300)
    color_map = get_model_color_map(df['model'].unique())
    for model_name, group in df.groupby("model"):
        label = normalize_model_label(model_name)
        error = np.abs(group['solubility_g_100g_log'] - group['prediction'])
        error_sorted = np.sort(error)
        cumulative = np.arange(1, len(error_sorted)+1) / len(error_sorted) * 100
        plt.plot(error_sorted, cumulative, label=label, color=color_map[model_name])
    plt.xlim(0, 4)
    plt.ylim(0, 100)
    plt.xlabel("Error, log(g/100g)")
    plt.ylabel("Cumulative % of predictions within error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_error_plot.png"))
    plt.close()

# === Combined Parity + Cumulative Layout ===
def plot_combined_summary(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    color_map = get_model_color_map(df['model'].unique())
    model_names = list(df['model'].unique())[:3]
    fig = plt.figure(figsize=(12, 10))
    spec = gridspec.GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[0, 1]), fig.add_subplot(spec[1, 0])]
    for ax, model in zip(axes, model_names):
        group = df[df['model'] == model]
        x = 10**group['solubility_g_100g_log']
        y = 10**group['prediction']
        label = normalize_model_label(model)
        ax.scatter(x, y, s=10, alpha=0.7, color=color_map[model])
        ax.plot([1e-3, 1e3], [1e-3, 1e3], color='black', linewidth=1)
        ax.plot([1e-3, 1e3], [1e-2, 1e4], color='green', linestyle='--', linewidth=0.7)
        ax.plot([1e-3, 1e3], [1e-4, 1e2], color='green', linestyle='--', linewidth=0.7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-3, 1e3)
        ax.set_ylim(1e-3, 1e3)
        ax.set_title(label)
        ax.axvline(1, color='black', linewidth=0.8)
        ax.axhline(1, color='black', linewidth=0.8)
        ax.set_xlabel("Experimental solubility, g/100g")
        ax.set_ylabel("Predicted solubility, g/100g")
    ax_cum = fig.add_subplot(spec[1, 1])
    for model_name, group in df.groupby("model"):
        label = normalize_model_label(model_name)
        error = np.abs(group['solubility_g_100g_log'] - group['prediction'])
        error_sorted = np.sort(error)
        cumulative = np.arange(1, len(error_sorted)+1) / len(error_sorted) * 100
        ax_cum.plot(error_sorted, cumulative, label=label, color=color_map[model_name])
    ax_cum.set_xlim(0, 4)
    ax_cum.set_ylim(0, 100)
    ax_cum.set_xlabel("Error, log(g/100g)")
    ax_cum.set_ylabel("Cumulative %")
    ax_cum.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_parity_cumulative.png"))
    plt.close()

# === Entry Point from main.py ===
def run_all_prediction_visualizations(df, output_dir, summary_csv_path):
    export_model_metrics(df, summary_csv_path)
    plot_combined_by_cv_type(df, output_dir)
    plot_individual_model_results(df, output_dir)
    plot_cumulative_error(df, output_dir)
    plot_combined_summary(df, output_dir)
