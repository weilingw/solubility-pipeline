import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required by mpl for 3D)
from matplotlib import colors as mcolors


def _role_descriptor_mapping(df, role, descriptor_type):
    """Return (logp_col, mw_col, tpsa_col) for a given role and descriptor family."""
    pref = f"{role}_"
    dt = (descriptor_type or "").lower()

    if dt == "rdkit":
        logp_col = pref + "MolLogP"  if pref + "MolLogP"  in df.columns else None
        mw_col   = pref + "MolWt"    if pref + "MolWt"    in df.columns else None
        tpsa_col = pref + "TPSA"     if pref + "TPSA"     in df.columns else None
    elif dt == "mordred":
        logp_col = pref + "SLogP"    if pref + "SLogP"    in df.columns else None
        mw_col   = pref + "MW"       if pref + "MW"       in df.columns else None
        tpsa_col = pref + "TopoPSA"  if pref + "TopoPSA"  in df.columns else None
    else:  # MOE-like fallback
        # common aliases
        candidates_logp = [pref + "LogP(o/w)", pref + "SlogP", pref + "logP", pref + "LogP"]
        logp_col = next((c for c in candidates_logp if c in df.columns), None)
        mw_col   = next((c for c in df.columns if c.startswith(pref) and "weight" in c.lower()), None)
        tpsa_col = next((c for c in df.columns if c.startswith(pref) and "tpsa"   in c.lower()), None)

    return logp_col, mw_col, tpsa_col


def _role_panel_2x2(
    df, role, descriptor_type, solubility_col, output_dir, model_type,
    descriptor_cols_role, base_font=12, marker_size=14, sol_norm=None, show_variance=False
):
    """
    Minimal, consistent 2×2 PCA panel for a single role (solute/solvent).
    Shows: Solubility, LogP-like, log10(MW), TPSA. No pair 2D plot included.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- PCA on role-specific descriptors ---
    X = df[descriptor_cols_role].astype(float)
    X_std = StandardScaler().fit_transform(X)
    X_std = pd.DataFrame(X_std, index=df.index, columns=descriptor_cols_role).fillna(0)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_std)
    evr = pca.explained_variance_ratio_

    pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=df.index)

    # axis labels (switch show_variance to True if want the percentages)
    xlab = f"PC1 ({evr[0]*100:.1f}%)" if show_variance else "PC1"
    ylab = f"PC2 ({evr[1]*100:.1f}%)" if show_variance else "PC2"

    # role-aware scalar properties
    logp_col, mw_col, tpsa_col = _role_descriptor_mapping(df, role, descriptor_type)

    vals = {
        "Solubility": df[solubility_col].values,
        "LogP":       (df[logp_col].values if logp_col else np.full(len(df), np.nan)),
        "logMolWt":   (np.log10(df[mw_col].replace(0, np.nan)) if mw_col is not None else np.full(len(df), np.nan)),
        "TPSA":       (df[tpsa_col].values if tpsa_col else np.full(len(df), np.nan)),
    }
    cbar_labels = {
        "Solubility": "Solubility (log scale)",
        "LogP":       (f"{logp_col} (LogP-like)" if logp_col else "Unavailable"),
        "logMolWt":   (f"{role.capitalize()} Molecular Weight (log₁₀ scale)" if mw_col else "Unavailable"),
        "TPSA":       (f"{tpsa_col}" if tpsa_col else "Unavailable"),
    }

    # symmetric limits for centered crosshair
    lim = float(np.nanmax([np.abs(pc_df["PC1"]).max(), np.abs(pc_df["PC2"]).max()]) or 1.0)
    margin = lim * 0.06
    xlo, xhi = -lim - margin, lim + margin
    ylo, yhi = -lim - margin, lim + margin

    # shared solubility normalisation if not provided
    if sol_norm is None:
        vmin, vmax = np.nanpercentile(df[solubility_col].values, [1, 99])
        sol_norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))

    # --- figure ---
    fig, ax_mat = plt.subplots(2, 2, figsize=(13.46, 11), dpi=600)  # RSC double column friendly
    axes = [ax_mat[0,0], ax_mat[0,1], ax_mat[1,0], ax_mat[1,1]]

    props = [
        ("Solubility", "viridis"),
        ("LogP",       "coolwarm"),
        ("logMolWt",   "plasma"),
        ("TPSA",       "cividis"),
    ]

    for (name, cmap), ax in zip(props, axes):
        arr = vals[name]
        norm = (sol_norm if name == "Solubility" else None)

        sc = ax.scatter(
            pc_df["PC1"], pc_df["PC2"],
            c=arr, cmap=cmap, norm=norm,
            s=marker_size, edgecolors="none", alpha=0.85
        )

        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.6)

        # ticks every ~10 units
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))

        # titles & labels
        title_map = {"logMolWt": "log10(MW)"}
        ax.set_title(title_map.get(name, name), fontsize=base_font+1, pad=8)
        ax.set_xlabel(xlab, fontsize=base_font)
        ax.set_ylabel(ylab, fontsize=base_font)
        ax.tick_params(axis="both", labelsize=base_font-2)

        # colorbar to the right of each panel
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.8%", pad=0.18)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.yaxis.set_label_position("right")
        cbar.set_label(cbar_labels[name], rotation=270, labelpad=14, fontsize=base_font-1)
        cbar.ax.tick_params(labelsize=base_font-3, length=3)

        # cleaner frame
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(wspace=0.24, hspace=0.28)
    out_base = f"{model_type.lower()}_{descriptor_type}_{role}_pca_panel"
    fig.savefig(os.path.join(output_dir, f"{out_base}.tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{out_base}.png"),  dpi=600, bbox_inches="tight")
    plt.close(fig)


def _pair_panel_combined_two_views_3d(
    df, pair_descriptor_cols, descriptor_type, solubility_col, output_dir, model_type,
    base_font=12, marker_size=10, sol_norm=None, show_variance=True
):
    """
    Pair-level PCA (combined solute_* + solvent_* descriptors) -> TWO 3D figures:
      - solute-view (colorbars sourced from solute_* columns where available)
      - solvent-view (colorbars sourced from solvent_* columns where available)
    This replaces the OLD 2D pair plot entirely.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- PCA (first 3 PCs) ---
    X = df[pair_descriptor_cols].astype(float)
    X_std = StandardScaler().fit_transform(X)
    X_std = pd.DataFrame(X_std, index=df.index, columns=pair_descriptor_cols).fillna(0)

    pca = PCA(n_components=min(3, X_std.shape[1]))
    pcs = pca.fit_transform(X_std)
    evr = pca.explained_variance_ratio_

    # light, distance-aware jitter (optional)
    if pcs.shape[0] > 10:
        try:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(8, len(pcs))).fit(pcs)
            dists, _ = nbrs.kneighbors(pcs)
            local = np.median(dists[:, 3:], axis=1)
            local = np.clip(local, np.quantile(local, 0.05), np.quantile(local, 0.95))
            rng = np.random.RandomState(0)
            pcs = pcs + rng.normal(size=pcs.shape) * (0.08 * local[:, None])
        except Exception:
            pass

    pc_names = [f"PC{i+1}" for i in range(pcs.shape[1])]
    axis_labels = [f"{n} ({evr[i]*100:.1f}%)" for i, n in enumerate(pc_names)] if show_variance else pc_names

    lim = float(np.nanmax(np.abs(pcs))) if pcs.size else 1.0
    margin = lim * 0.06
    lo, hi = -lim - margin, lim + margin

    # shared solubility norm
    if sol_norm is None:
        vmin, vmax = np.nanpercentile(df[solubility_col].values, [1, 99])
        sol_norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))

    def _role_vals_and_labels(role):
        logp, mw, tpsa = _role_descriptor_mapping(df, role, descriptor_type)
        vals = {
            "Solubility": df[solubility_col].values,
            "LogP":       (df[logp].values if logp else np.full(len(df), np.nan)),
            "logMolWt":   (np.log10(df[mw].replace(0, np.nan)) if mw else np.full(len(df), np.nan)),
            "TPSA":       (df[tpsa].values if tpsa else np.full(len(df), np.nan)),
        }
        cbar_labels = {
            "Solubility": "Solubility (log scale)",
            "LogP":       (f"{logp} (LogP-like)" if logp else "Unavailable"),
            "logMolWt":   (f"{role.capitalize()} Molecular Weight (log₁₀ scale)" if mw else "Unavailable"),
            "TPSA":       (f"{tpsa}" if tpsa else "Unavailable"),
        }
        return vals, cbar_labels

    def _plot_view(role_tag, vals, cbar_labels, out_tag):
        fig = plt.figure(figsize=(11.0, 9.0), dpi=600)
        axes = [
            fig.add_subplot(2, 2, 1, projection='3d'),
            fig.add_subplot(2, 2, 2, projection='3d'),
            fig.add_subplot(2, 2, 3, projection='3d'),
            fig.add_subplot(2, 2, 4, projection='3d'),
        ]
        props = [
            ("Solubility", "viridis"),
            ("LogP",       "coolwarm"),
            ("logMolWt",   "plasma"),
            ("TPSA",       "cividis"),
        ]

        order = np.argsort(pcs[:, 2])  # back→front
        Xp, Yp, Zp = pcs[order, 0], pcs[order, 1], pcs[order, 2]

        for (name, cmap), ax in zip(props, axes):
            arr = vals[name][order]
            norm = (sol_norm if name == "Solubility" else None)

            sc = ax.scatter(
                Xp, Yp, Zp, c=arr, cmap=cmap, norm=norm,
                s=marker_size*0.55, alpha=0.55, edgecolors="none", depthshade=False
            )

            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
            ax.plot([lo, hi],[0,0],[0,0],'k--',lw=0.6)
            ax.plot([0,0],[lo,hi],[0,0],'k--',lw=0.6)
            ax.plot([0,0],[0,0],[lo,hi],'k--',lw=0.6)

            ax.set_xlabel(axis_labels[0], labelpad=6, fontsize=base_font)
            ax.set_ylabel(axis_labels[1], labelpad=6, fontsize=base_font)
            ax.set_zlabel(axis_labels[2], labelpad=6, fontsize=base_font)
            ax.zaxis.set_rotate_label(False)
            ax.zaxis.set_label_coords(-0.12, 0.5)
            ax.zaxis.set_tick_params(pad=2)
            for lbl in ax.zaxis.get_ticklabels():
                lbl.set_horizontalalignment("left")

            ax.xaxis.set_major_locator(MultipleLocator(15))
            ax.yaxis.set_major_locator(MultipleLocator(15))
            ax.zaxis.set_major_locator(MultipleLocator(15))

            ax.set_title("log10(MW)" if name == "logMolWt" else name, fontsize=base_font+1, pad=6)

            cbar = fig.colorbar(sc, ax=ax, pad=0.18, shrink=0.78)
            cbar.set_label(cbar_labels[name], rotation=270, labelpad=18, fontsize=base_font-1)
            cbar.ax.tick_params(labelsize=base_font-3, length=3)

        fig.subplots_adjust(wspace=0.10, hspace=0.16)
        out_base = f"{model_type.lower()}_{descriptor_type}_pair3d_combined_{out_tag}_pca_panel"
        fig.savefig(os.path.join(output_dir, f"{out_base}.tiff"), dpi=600, bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, f"{out_base}.png"),  dpi=600, bbox_inches="tight")
        plt.close(fig)

    sol_vals, sol_lbls = _role_vals_and_labels("solute")
    _plot_view("solute", sol_vals, sol_lbls, out_tag="soluteview")

    solv_vals, solv_lbls = _role_vals_and_labels("solvent")
    _plot_view("solvent", solv_vals, solv_lbls, out_tag="solventview")


# --- PUBLIC API -----------------------------------------------------------------
def compare_descriptor_pca(
    df,
    descriptor_cols,
    descriptor_type,
    solubility_col,
    solute_name_col,   # kept for signature compatibility; not used below
    output_dir,
    model_type,
    role_prefix="solute",   # which role to read LogP/MW/TPSA from: "solute" or "solvent"
    show_variance=False,    # if True: axis labels include explained variance
    pair_3d=True, 
):
    # --- PCA on the PAIR-LEVEL descriptor matrix provided ---
    X = df[descriptor_cols].astype(float)
    X_std = StandardScaler().fit_transform(X)
    X_std = pd.DataFrame(X_std, index=df.index, columns=descriptor_cols).fillna(0)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_std)
    evr = pca.explained_variance_ratio_

    pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=df.index)
    pc_df["Solubility"] = df[solubility_col].values

    # --- pick columns for coloring (from the chosen role) ---
    pref = f"{role_prefix}_"
    dt = (descriptor_type or "").lower()

    if dt == "rdkit":
        logp_col = pref + "MolLogP"  if pref + "MolLogP"  in df.columns else None
        mw_col   = pref + "MolWt"    if pref + "MolWt"    in df.columns else None
        tpsa_col = pref + "TPSA"     if pref + "TPSA"     in df.columns else None
    elif dt == "mordred":
        logp_col = pref + "SLogP"    if pref + "SLogP"    in df.columns else None
        mw_col   = pref + "MW"       if pref + "MW"       in df.columns else None
        tpsa_col = pref + "TopoPSA"  if pref + "TopoPSA"  in df.columns else None
    else:  # MOE-like fallback
        candidates_logp = [pref + "LogP(o/w)", pref + "SlogP", pref + "logP", pref + "LogP"]
        logp_col = next((c for c in candidates_logp if c in df.columns), None)
        mw_col   = next((c for c in df.columns if c.startswith(pref) and "weight" in c.lower()), None)
        tpsa_col = next((c for c in df.columns if c.startswith(pref) and "tpsa"   in c.lower()), None)

    # Labels
    mollogp_label = f"{logp_col} (LogP-like)" if logp_col else "Unavailable"
    molwt_label   = f"{role_prefix.capitalize()} Molecular Weight (log₁₀ scale)" if mw_col else "Unavailable"
    tpsa_label    = f"{tpsa_col}" if tpsa_col else "Unavailable"

    # Values for coloring
    pc_df["MolLogP"]  = df[logp_col].values if logp_col else np.full(len(df), np.nan)
    pc_df["logMolWt"] = (np.log10(df[mw_col].replace(0, np.nan)) if mw_col else np.full(len(df), np.nan))
    pc_df["TPSA"]     = df[tpsa_col].values if tpsa_col else np.full(len(df), np.nan)

    # Shared limits for a centered crosshair
    lim = float(max(
        abs(np.nanmax(pc_df["PC1"])), abs(np.nanmin(pc_df["PC1"])),
        abs(np.nanmax(pc_df["PC2"])), abs(np.nanmin(pc_df["PC2"]))
    ) or 1.0)
    margin = lim * 0.05

    # Axis labels
    xlab = f"PC1 ({evr[0]*100:.1f}%)" if show_variance else "PC1"
    ylab = f"PC2 ({evr[1]*100:.1f}%)" if show_variance else "PC2"

    # Use robust, built-in matplotlib colormaps
    properties = [
        ("Solubility", "Solubility (log scale)", "viridis"),
        ("MolLogP",    mollogp_label,             "coolwarm"),
        ("logMolWt",   molwt_label,               "plasma"),
        ("TPSA",       tpsa_label,                "cividis"),
    ]

    # Shared solubility normalization so panels are comparable
    sol_vals = pc_df["Solubility"].values
    if np.isfinite(sol_vals).any():
        vmin, vmax = np.nanpercentile(sol_vals, [1, 99])
    else:
        vmin, vmax = float(np.nanmin(sol_vals)), float(np.nanmax(sol_vals))
    sol_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13.46, 11), dpi=600)
    axes = axes.flatten()

    if pair_3d:
        pair_cols = [c for c in descriptor_cols if c.startswith(("solute_", "solvent_"))] or list(descriptor_cols)
        _pair_panel_combined_two_views_3d(
            df, pair_cols, descriptor_type, solubility_col, output_dir, model_type,
            base_font=12, marker_size=10, sol_norm=sol_norm, show_variance=show_variance
        )

    for ax, (col, label, cmap) in zip(axes, properties):
        vals = pc_df[col].values
        if np.all(np.isnan(vals)):
            # Plot anyway with zeros, and annotate
            vals = np.zeros_like(pc_df["PC1"].values)
            print(f"⚠️ {col} is entirely NaN — plotting zeros for visual continuity.")

        norm = sol_norm if col == "Solubility" else None

        sc = ax.scatter(
            pc_df["PC1"], pc_df["PC2"],
            c=vals, cmap=cmap, norm=norm,
            s=20, edgecolors='none', alpha=0.85
        )

        ax.set_xlim(-lim - margin, lim + margin)
        ax.set_ylim(-lim - margin, lim + margin)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.6)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.6)
        ax.set_xlabel(xlab, fontsize=10, labelpad=6)
        ax.set_ylabel(ylab, fontsize=10, labelpad=6)
        title_map = {"logMolWt": "log10(MW)"}
        ax.set_title(f"PCA colored by {title_map.get(col, label)}", fontsize=11, pad=10)
        ax.tick_params(axis='both', labelsize=8)

        cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
        cbar.set_label(label, fontsize=9)

        # clean frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    plt.tight_layout()
    out_base = f"{model_type.lower()}_{descriptor_type}_pair2d_pca_panel"
    fig.savefig(os.path.join(output_dir, f"{out_base}.tiff"), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f"{out_base}.png"),  dpi=600, bbox_inches='tight')
    plt.close(fig)
