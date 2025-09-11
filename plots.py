# prediction_plot_utils.py

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.lines import Line2D

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
    print(f"✅ Model evaluation metrics saved to: {output_path}")

# === Combined Parity Plots by CV Type ===
def plot_combined_by_cv_type(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for cv_type, group in df.groupby("cv_type"):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        # scatter per model (same color = cv color)
        for model in group['model'].unique():
            model_df = group[group['model'] == model]
            ax.scatter(
                model_df['solubility_g_100g_log'],
                model_df['prediction'],
                label=normalize_model_label(model),
                alpha=0.7,
                color=cv_colors.get(cv_type, "gray"),
            )

        min_val = min(group['solubility_g_100g_log'].min(), group['prediction'].min()) - 0.5
        max_val = max(group['solubility_g_100g_log'].max(), group['prediction'].max()) + 0.5
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
        ax.plot([min_val, max_val], [min_val + 1, max_val + 1], 'r--', lw=0.75, label='±1 log unit')
        ax.plot([min_val, max_val], [min_val - 1, max_val - 1], 'r--', lw=0.75)

        ax.set_title(f'Combined Parity Plot — {cv_type} CV')
        ax.set_xlabel('True solubility (log g/100g)')
        ax.set_ylabel('Predicted solubility (log g/100g)')
        ax.grid(True)

        # legend for models (text labels) …
        ax.legend(title="Models (all in CV color)", loc="lower right")

        # …and an additional legend explaining the CV color itself
        cv_handle = [Line2D([0],[0], marker='o', color='w', label=f"{cv_type} models",
                            markerfacecolor=cv_colors.get(cv_type, "gray"), markersize=8)]
        ax.add_artist(ax.legend(handles=cv_handle, title="CV Type", loc="upper left"))

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{cv_type}_combined_parity.png"))
        plt.close(fig)

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
    plt.xlim(0, 4); plt.ylim(0, 100)
    plt.title("Cumulative error (all models / CV types)")
    plt.xlabel("Error, log(g/100g)")
    plt.ylabel("Cumulative % of predictions within error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_error_plot.png"))
    plt.close()


def plot_tuned_vs_cv_with_cosmo(df, output_path, model_regex=None, title=None):
    """
    Make a 2x2 figure:
      [0,0] tuned model — LOSO (log–log parity)
      [0,1] tuned model — 10-fold (log–log parity)
      [1,0] COSMO-RS baseline (log–log parity)
      [1,1] cumulative error curves (all three)

    Parameters
    ----------
    df : pd.DataFrame
        Needs columns: ['solubility_g_100g_log','prediction','model','cv_type'].
    output_path : str
        Where to save (e.g., os.path.join(output_dir, "combined_tuned_cv_cosmo.png")).
    model_regex : str or None
        A regex to select the tuned model label (e.g., r"xgb.*hybrid.*tuned").
        If None, we auto-pick the most frequent tuned model in df.
    title : str or None
        Optional suptitle for the whole figure.
    """

    # --- pick the tuned model you want to compare ---
    tuned_mask = df['model'].str.contains('tuned', case=False, na=False)
    tuned_df = df[tuned_mask].copy()

    if tuned_df.empty:
        raise ValueError("No tuned models found in df['model'].")

    if model_regex is None:
        # choose the most frequent tuned model name
        chosen_model = tuned_df['model'].value_counts().index[0]
    else:
        candidates = tuned_df['model'].unique()
        matches = [m for m in candidates if _re.search(model_regex, m, flags=_re.IGNORECASE)]
        if not matches:
            raise ValueError(f"No tuned model matches regex: {model_regex!r}. "
                             f"Available tuned models: {list(candidates)}")
        chosen_model = matches[0]

    # --- slice the three series we need ---
    def _slice(model_name, cv):
        return df[(df['model'] == model_name) & (df['cv_type'].str.upper() == cv.upper())]

    loso = _slice(chosen_model, 'LOSO')
    k10  = _slice(chosen_model, '10Fold')
    cosmo = df[df['cv_type'].str.upper() == 'COSMO']  # baseline rows

    if loso.empty:
        raise ValueError(f"No rows for chosen tuned model under LOSO: {chosen_model}")
    if k10.empty:
        raise ValueError(f"No rows for chosen tuned model under 10Fold: {chosen_model}")
    if cosmo.empty:
        raise ValueError("No COSMO baseline rows found (cv_type == 'COSMO').")

    # --- colors (consistent with your scheme) ---
    color_map = get_model_color_map([chosen_model, "COSMO-RS (Baseline)"])
    c_model = color_map.get(chosen_model, '#FF6F61')     # model color
    c_cosmo = color_map.get("COSMO-RS (Baseline)", '#5B9BD5')

    # --- plotting helpers ---
    def _parity_loglog(ax, group, label, color):
        x = 10**group['solubility_g_100g_log'].to_numpy()
        y = 10**group['prediction'].to_numpy()
        ax.scatter(x, y, s=12, alpha=0.75, color=color, edgecolors='none')
        # diagonal and ±1 log decade bands
        ax.plot([1e-3, 1e3], [1e-3, 1e3], color='black', linewidth=1)
        ax.plot([1e-3, 1e3], [1e-2, 1e4], color='green', linestyle='--', linewidth=0.7, label="±1 log unit")
        ax.plot([1e-3, 1e3], [1e-4, 1e2], color='green', linestyle='--', linewidth=0.7)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(1e-3, 1e3); ax.set_ylim(1e-3, 1e3)
        ax.axvline(1, color='black', linewidth=0.8); ax.axhline(1, color='black', linewidth=0.8)
        ax.set_xlabel("Experimental solubility, g/100g")
        ax.set_ylabel("Predicted solubility, g/100g")
        ax.set_title(label)

    def _cum_err(ax, label_color_pairs):
        for label, color, group in label_color_pairs:
            err = _np.abs(group['solubility_g_100g_log'] - group['prediction'])
            srt = _np.sort(err)
            cum = _np.arange(1, len(srt)+1) / len(srt) * 100
            ax.plot(srt, cum, label=label, color=color)
        ax.set_xlim(0, 4); ax.set_ylim(0, 100)
        ax.set_xlabel("Error, log(g/100g)")
        ax.set_ylabel("Cumulative %")
        ax.legend()

    # --- build figure ---
    fig = plt.figure(figsize=(12, 10), dpi=300)
    gs = fig.add_gridspec(2, 2)

    ax11 = fig.add_subplot(gs[0, 0])  # tuned LOSO
    ax12 = fig.add_subplot(gs[0, 1])  # tuned 10-fold
    ax21 = fig.add_subplot(gs[1, 0])  # COSMO baseline
    ax22 = fig.add_subplot(gs[1, 1])  # cumulative error

    _parity_loglog(ax11, loso,  f"{normalize_model_label(chosen_model)} (LOSO)", c_model)
    _parity_loglog(ax12, k10,   f"{normalize_model_label(chosen_model)} (10-fold)", c_model)
    _parity_loglog(ax21, cosmo, "cosmo-RS (Baseline)", c_cosmo)

    _cum_err(ax22, [
        (f"cosmo-RS (Baseline)", c_cosmo, cosmo),
        (f"{normalize_model_label(chosen_model)} (LOSO)", c_model, loso),
        (f"{normalize_model_label(chosen_model)} (10-fold)", c_model, k10),
    ])

    if title:
        fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    ax22.set_title("Cumulative error")
    fig.savefig(output_path)
    plt.close(fig)

# === Entry Point from main.py ===
def run_all_prediction_visualizations(df, output_dir, summary_csv_path):
    export_model_metrics(df, summary_csv_path)
    plot_combined_by_cv_type(df, output_dir)
    plot_individual_model_results(df, output_dir)
    plot_cumulative_error(df, output_dir)
