# ------------------- pca_rdkit.py (both settings in one code) -------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors



# --- helper: make one panel for a role ("solute"/"solvent") in chosen row-mode ---
def _role_panel(
    df, role, descriptor_type, solubility_col, output_dir, model_type,
    descriptor_cols_role,                # list of columns for this role (either solute_* or solvent_*)
    rows_mode="pair",                    # "pair" (all rows) or "entity" (unique {role})
    base_font=12, marker_size=14, sol_norm=None
):
    """
    rows_mode="pair":  use ALL rows (pairs); many will share identical coords -> jitter helps
    rows_mode="entity": aggregate to one row per unique {role}_smiles with median solubility; size=freq
    """

    os.makedirs(output_dir, exist_ok=True)

    pref = f"{role}_"
    dt = (descriptor_type or "").lower()
    # role-aware optional properties for side panels
    if dt == "rdkit":
        logp_col = pref + "MolLogP"  if pref + "MolLogP"  in df.columns else None
        mw_col   = pref + "MolWt"    if pref + "MolWt"    in df.columns else None
        tpsa_col = pref + "TPSA"     if pref + "TPSA"     in df.columns else None
    elif dt == "mordred":
        logp_col = pref + "SLogP"    if pref + "SLogP"    in df.columns else None
        mw_col   = pref + "MW"       if pref + "MW"       in df.columns else None
        tpsa_col = pref + "TopoPSA"  if pref + "TopoPSA"  in df.columns else None
    else:  # MOE-like
        logp_col = pref + "LogP(o/w)"    if pref + "LogP(o/w)"    in df.columns else None
        mw_col   = next((c for c in df.columns if c.startswith(pref) and "weight" in c.lower()), None)
        tpsa_col = next((c for c in df.columns if c.startswith(pref) and "tpsa"   in c.lower()), None)

    # choose plotting dataframe
    key_col = f"{role}_smiles" if f"{role}_smiles" in df.columns else None
    if rows_mode == "entity" and key_col:
        stats = df.groupby(key_col)[solubility_col].agg(
            n="size", med="median",
            q1=lambda x: np.nanpercentile(x, 25),
            q3=lambda x: np.nanpercentile(x, 75)
        )
        df_plot = df.drop_duplicates(subset=[key_col]).copy()
        df_plot = df_plot.merge(stats, left_on=key_col, right_index=True, how="left")
        df_plot["__freq__"] = df_plot["n"].astype(int)
        color_sol = df_plot["med"].values  # median solubility
    else:
        df_plot = df.copy()
        if key_col:
            df_plot["__freq__"] = df_plot[key_col].map(df[key_col].value_counts()).astype(int)
        else:
            df_plot["__freq__"] = 1
        color_sol = df_plot[solubility_col].values

    # PCA on the role-specific descriptors
    X = df_plot[descriptor_cols_role].astype(float)
    X_std = StandardScaler().fit_transform(X)
    X_std = pd.DataFrame(X_std, index=df_plot.index, columns=descriptor_cols_role).fillna(0)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_std)
    evr = pca.explained_variance_ratio_

    pc_df = pd.DataFrame(pcs, columns=["PC1","PC2"], index=df_plot.index)
    # auxiliary values for colouring
    vals = {
        "Solubility": color_sol,
        "LogP":    (df_plot[logp_col].values if logp_col else np.full(len(df_plot), np.nan)),
        "logMolWt":   (np.log10(df_plot[mw_col].replace(0, np.nan)) if mw_col else np.full(len(df_plot), np.nan)),
        "TPSA":       (df_plot[tpsa_col].values if tpsa_col else np.full(len(df_plot), np.nan)),
    }
    cbar_labels = {
        "Solubility": "Solubility (log scale)",
        "LogP":    (f"{logp_col} (LogP-like)" if logp_col else "Unavailable"),
        "logMolWt":   (f"{role.capitalize()} Molecular Weight (log₁₀ scale)" if mw_col else "Unavailable"),
        "TPSA":       (f"{tpsa_col} (Topological Polar Surface Area)" if tpsa_col else "Unavailable"),
    }

    # symmetric limits & jitter (helps in pair mode)
    lim = float(np.nanmax([np.abs(pc_df["PC1"]).max(), np.abs(pc_df["PC2"]).max()]) or 1.0)
    margin = lim * 0.06
    xlo, xhi = -lim - margin, lim + margin
    ylo, yhi = -lim - margin, lim + margin

    rng = np.random.RandomState(0)
    span = (xhi - xlo)
    j = span * (0.004 if rows_mode == "pair" else 0.0)
    PC1j = pc_df["PC1"].values + (rng.normal(0, j, len(pc_df)) if j > 0 else 0)
    PC2j = pc_df["PC2"].values + (rng.normal(0, j, len(pc_df)) if j > 0 else 0)

    # build figure (2×2 panels)
    fig, ax_mat = plt.subplots(2, 2, figsize=(2.08, 1.57), dpi=600)
    axes = [ax_mat[0,0], ax_mat[0,1], ax_mat[1,0], ax_mat[1,1]]

    props = [
        ("Solubility", "viridis"),
        ("LogP",    "coolwarm"),
        ("logMolWt",   "plasma"),
        ("TPSA",       "cividis"),
    ]
    xlab = "PC1"; ylab = "PC2"
    #xlab = f"PC1 ({evr[0]*100:.1f}%)"
    #ylab = f"PC2 ({evr[1]*100:.1f}%)"

    # shared solubility normalization (pass in or compute once overall)
    if sol_norm is None:
        vmin, vmax = np.nanpercentile(df[solubility_col].values, [1, 99])
        sol_norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))

    # size scaling by frequency (more visible in entity mode; also ok in pair mode)
    f = np.sqrt(np.asarray(df_plot["__freq__"], dtype=float))
    size_scale = np.clip(1 + 0.25*(f - 1), 1, 2.2)
    s_val = marker_size * 0.5 * size_scale

    for k, ((name, cmap), ax) in enumerate(zip(props, axes)):
        arr = vals[name]
        norm = (sol_norm if name == "Solubility" else None)

        sc = ax.scatter(PC1j, PC2j, c=arr, cmap=cmap, norm=norm,
                        s=s_val, alpha=0.55 if rows_mode=="pair" else 0.8,
                        edgecolors="none")

        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # ticks: every 10; hide top-row x and right-col y labels
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        if k < 2:      ax.tick_params(labelbottom=False)
        if k % 2 == 1: ax.tick_params(labelleft=False)

        # simple, centered titles (no model name)
        title_map = {"logMolWt": "log10(MW)"}
        ax.set_title(title_map.get(name, name), fontsize=base_font+1, loc="center")
        ax.set_xlabel(""); ax.set_ylabel("")

        # full-height colorbar on the right
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.16)  # a bit wider + more gap
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.yaxis.set_label_position("right")
        cbar.set_label(cbar_labels[name], rotation=270, labelpad=18, fontsize=base_font)
        cbar.ax.tick_params(labelsize=base_font-2, length=3)

    # global labels (no model name in the title)
    fig.supxlabel(xlab, fontsize=base_font+1, y=0.04)
    fig.supylabel(ylab, fontsize=base_font+1, x=0.04)
    fig.subplots_adjust(right=0.92)

    # file base: includes rows_mode and role; figure itself has no model name
    out_base = f"{model_type.lower()}_{descriptor_type}_{rows_mode}_{role}_pca_panel"
    fig.savefig(os.path.join(output_dir, f"{out_base}.tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{out_base}.png"),  dpi=600, bbox_inches="tight")
    plt.close(fig)

def _pair_panel_combined_two_views_3d(
    df, pair_descriptor_cols, descriptor_type, solubility_col, output_dir, model_type,
    base_font=12, marker_size=10, sol_norm=None, show_variance=True
):
    """
    Pair-level PCA on combined solute_* + solvent_* descriptors.
    Produces TWO 3D figures (same PCs): solute-view & solvent-view.
    No jitter, no frequency sizing.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    # --- PCA (take first 3 PCs) ---
    X = df[pair_descriptor_cols].astype(float)
    X_std = StandardScaler().fit_transform(X)
    X_std = pd.DataFrame(X_std, index=df.index, columns=pair_descriptor_cols).fillna(0)

    pca = PCA(n_components=min(3, X_std.shape[1]))
    pcs = pca.fit_transform(X_std)
    evr = pca.explained_variance_ratio_

    enable_jitter = True
    if enable_jitter and pcs.shape[0] > 10:
        try:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(8, len(pcs))).fit(pcs)
            dists, _ = nbrs.kneighbors(pcs)
            local = np.median(dists[:, 3:], axis=1)  # ignore the first few tiny distances
            local = np.clip(local, np.quantile(local, 0.05), np.quantile(local, 0.95))
            rng = np.random.RandomState(0)
            pcs = pcs + rng.normal(size=pcs.shape) * (0.08 * local[:, None])  # ~8% of local scale
        except Exception:
            pass
    # names & labels
    pc_names = [f"PC{i+1}" for i in range(pcs.shape[1])]
    if show_variance:
        axis_labels = [f"{n} ({evr[i]*100:.1f}%)" for i, n in enumerate(pc_names)]
    else:
        axis_labels = pc_names

    # symmetric limits across all three axes so (0,0,0) is centered
    lim = float(np.nanmax(np.abs(pcs))) if pcs.size else 1.0
    margin = lim * 0.06
    lo, hi = -lim - margin, lim + margin

    # shared solubility norm
    if sol_norm is None:
        vmin, vmax = np.nanpercentile(df[solubility_col].values, [1, 99])
        sol_norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))

    # role-specific color variables
    def _role_vals_and_labels(role):
        pref = f"{role}_"
        dt = (descriptor_type or "").lower()
        if dt == "rdkit":
            logp = pref + "MolLogP"  if pref + "MolLogP"  in df.columns else None
            mw   = pref + "MolWt"    if pref + "MolWt"    in df.columns else None
            tpsa = pref + "TPSA"     if pref + "TPSA"     in df.columns else None
        elif dt == "mordred":
            logp = pref + "SLogP"    if pref + "SLogP"    in df.columns else None
            mw   = pref + "MW"       if pref + "MW"       in df.columns else None
            tpsa = pref + "TopoPSA"  if pref + "TopoPSA"  in df.columns else None
        else:
            logp = pref + "SlogP"    if pref + "SlogP"    in df.columns else None
            mw   = next((c for c in df.columns if c.startswith(pref) and "weight" in c.lower()), None)
            tpsa = next((c for c in df.columns if c.startswith(pref) and "tpsa"   in c.lower()), None)

        vals = {
            "Solubility": df[solubility_col].values,
            "LogP":    (df[logp].values if logp else np.full(len(df), np.nan)),
            "logMolWt":   (np.log10(df[mw].replace(0, np.nan)) if mw else np.full(len(df), np.nan)),
            "TPSA":       (df[tpsa].values if tpsa else np.full(len(df), np.nan)),
        }
        cbar_labels = {
            "Solubility": "Solubility (log scale)",
            "LogP":    (f"{logp} (LogP-like)" if logp else "Unavailable"),
            "logMolWt":   (f"{role.capitalize()} Molecular Weight (log₁₀ scale)" if mw else "Unavailable"),
            "TPSA":       (f"{tpsa} (Topological Polar Surface Area)" if tpsa else "Unavailable"),
        }
        return vals, cbar_labels

    def _plot_view(role_tag, vals, cbar_labels, out_tag):
        # 2×2 grid of 3D subplots
        fig = plt.figure(figsize=(11.0, 9.0), dpi=600)
        axes = [
            fig.add_subplot(2, 2, 1, projection='3d'),
            fig.add_subplot(2, 2, 2, projection='3d'),
            fig.add_subplot(2, 2, 3, projection='3d'),
            fig.add_subplot(2, 2, 4, projection='3d'),
        ]
        props = [
            ("Solubility", "viridis"),
            ("LogP",    "coolwarm"),
            ("logMolWt",   "plasma"),
            ("TPSA",       "cividis"),
        ]

        for k, ((name, cmap), ax) in enumerate(zip(props, axes)):
            arr = vals[name]
            norm = (sol_norm if name == "Solubility" else None)

            # draw back→front by sorting on Z
            order = np.argsort(pcs[:, 2])
            Xp, Yp, Zp = pcs[order, 0], pcs[order, 1], pcs[order, 2]
            arr_sorted = arr[order]

            sc = ax.scatter(
                Xp, Yp, Zp,
                c=arr_sorted, cmap=cmap, norm=norm,
                s=marker_size*0.55,   # smaller dots
                alpha=0.55,           # a bit transparent
                edgecolors="none",
                depthshade=False
            )

            # limits + origin cross
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
            ax.plot([lo, hi],[0,0],[0,0],'k--',lw=0.6)
            ax.plot([0,0],[lo,hi],[0,0],'k--',lw=0.6)
            ax.plot([0,0],[0,0],[lo,hi],'k--',lw=0.6)

            # labels (optionally without variance)
            ax.set_xlabel(axis_labels[0], labelpad=6, fontsize=base_font)
            ax.set_ylabel(axis_labels[1], labelpad=6, fontsize=base_font)
            ax.set_zlabel(axis_labels[2], labelpad=6, fontsize=base_font)

            # --- put Z label on the left side to avoid cbar overlap ---
            ax.zaxis.set_rotate_label(False)           # don’t auto-rotate
            ax.zaxis.set_label_coords(-0.12, 0.5)      # <- move label to left (axes fraction)
            ax.zaxis.set_tick_params(pad=2)            # keep tick labels close to axis
            for lbl in ax.zaxis.get_ticklabels():
                lbl.set_horizontalalignment("left")    # visually left-aligned


            # ticks every 10
            ax.xaxis.set_major_locator(MultipleLocator(15))
            ax.yaxis.set_major_locator(MultipleLocator(15))
            ax.zaxis.set_major_locator(MultipleLocator(15))

            # title
            ax.set_title("log10(MW)" if name=="logMolWt" else name,
                         fontsize=base_font+1, loc="center")

            # colorbar (simple placement works better with 3D)
            cbar = fig.colorbar(sc, ax=ax, pad=0.18, shrink=0.78)
            cbar.set_label(cbar_labels[name], rotation=270, labelpad=18, fontsize=base_font)
            cbar.ax.tick_params(labelsize=base_font-2, length=3)

        fig.subplots_adjust(wspace=0.10, hspace=0.16)
        out_base = f"{model_type.lower()}_{descriptor_type}_pair3d_combined_{out_tag}_pca_panel"
        fig.savefig(os.path.join(output_dir, f"{out_base}.tiff"), dpi=600, bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, f"{out_base}.png"),  dpi=600, bbox_inches="tight")
        plt.close(fig)

    # make the two views with the SAME 3D PCs
    sol_vals, sol_lbls = _role_vals_and_labels("solute")
    _plot_view("solute", sol_vals, sol_lbls, out_tag="soluteview")

    solv_vals, solv_lbls = _role_vals_and_labels("solvent")
    _plot_view("solvent", solv_vals, solv_lbls, out_tag="solventview")


# --- PUBLIC: produce FOUR figures ------------------------------------------------
def compare_descriptor_pca(
    df,
    descriptor_cols,
    descriptor_type,
    solubility_col,
    solute_name_col,
    output_dir,
    model_type,
    base_font=12,
    marker_size=14,
    pair_3d=True,           # <<< NEW: render pair-level as 3D combined PCA
    show_variance=False     # <<< NEW: hide variance percentages in axis labels if False
):
    os.makedirs(output_dir, exist_ok=True)

    solute_cols  = [c for c in descriptor_cols if c.startswith("solute_")]
    solvent_cols = [c for c in descriptor_cols if c.startswith("solvent_")]
    pair_cols    = [c for c in descriptor_cols if c.startswith(("solute_","solvent_"))]

    vmin, vmax = np.nanpercentile(df[solubility_col].values, [1, 99])
    sol_norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))

    # 1–2) PAIR-LEVEL (combined) — 3D or 2D depending on flag
    if pair_cols:
        if pair_3d:
            _pair_panel_combined_two_views_3d(
                df, pair_cols, descriptor_type, solubility_col, output_dir, model_type,
                base_font=base_font, marker_size=marker_size, sol_norm=sol_norm,
                show_variance=show_variance
            )
        else:
            # if still want the 2D combined pair figures, call the 2D function here
            pass
    else:
        print("⚠️ No pair-level descriptor columns (solute_* / solvent_).")

    # 3) ENTITY-LEVEL — SOLUTE
    if solute_cols:
        _role_panel(df, "solute", descriptor_type, solubility_col, output_dir, model_type,
                    solute_cols, rows_mode="entity",
                    base_font=base_font, marker_size=marker_size, sol_norm=sol_norm)

    # 4) ENTITY-LEVEL — SOLVENT
    if solvent_cols:
        _role_panel(df, "solvent", descriptor_type, solubility_col, output_dir, model_type,
                    solvent_cols, rows_mode="entity",
                    base_font=base_font, marker_size=marker_size, sol_norm=sol_norm)


