import os, re, glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")               # set backend BEFORE importing pyplot
import matplotlib.pyplot as plt

MODELS = ["rf", "xgb", "svm"]
DESCS  = ["morgan", "moe", "mordred", "rdkit"]

# e.g. xgb_moe_hybrid_tuned_kfold_r2_scrambling.csv
TAG_RE = re.compile(r"(rf|xgb|svm)_(morgan|moe|mordred|rdkit)_.+_(kfold|loso)_r2_scrambling\.csv$", re.I)


def _collect_records(scramble_root: str):
    recs = []
    print(f"[scan] root = {scramble_root}")
    for path in glob.glob(os.path.join(scramble_root, "**", "*_r2_scrambling.csv"), recursive=True):
        fname = os.path.basename(path)
        print(f"[found] {fname}")
        m = TAG_RE.search(fname)
        if not m:
            print(f"[skip-regex] {fname} did not match TAG_RE")
            continue
        model = m.group(1).lower()
        desc  = m.group(2).lower()
        cv    = "10-fold" if m.group(3).lower() == "kfold" else "LOSO"

        try:
            dfp = pd.read_csv(path)
            # tolerate column naming variants
            perm_col = "R2_perm" if "R2_perm" in dfp.columns else ("perm_R2" if "perm_R2" in dfp.columns else None)
            if perm_col is None:
                print(f"[skip-cols] {fname} missing perm column (R2_perm/perm_R2). cols={list(dfp.columns)}")
                continue
            obs_col = "obs_R2" if "obs_R2" in dfp.columns else ("obs_r2" if "obs_r2" in dfp.columns else None)
            p_col   = "p_value" if "p_value" in dfp.columns else ("pval" if "pval" in dfp.columns else None)
            if obs_col is None or p_col is None:
                print(f"[skip-cols] {fname} missing obs/p columns. cols={list(dfp.columns)}")
                continue

            recs.append(dict(
                model=model, desc=desc, cv=cv,
                obs_R2=float(dfp[obs_col].iloc[0]),
                p_value=float(dfp[p_col].iloc[0]),
                perm_R2=dfp[perm_col].to_numpy(),
                path=path,
            ))
            print(f"[ok] key=({model},{desc},{cv}) n_perm={len(dfp[perm_col])}")
        except Exception as e:
            print(f"[error] {fname}: {e}")
            continue
    print(f"[summary] collected {len(recs)} records")
    return recs

def _plot_matrix(by_key, models, descs, cv_label, outfile):
    fig, axes = plt.subplots(len(models), len(descs),
                             figsize=(7, 4.5),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    # keep titles small and light so they don’t look “double bold”
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.titleweight'] = 'regular'
    plt.rcParams['axes.titlepad'] = 2

    PANEL_FS = 9
    TICK_FS  = 8
    ANNOT_FS = 9

    for i, m in enumerate(models):
        for j, d in enumerate(descs):
            ax = axes[i, j]
            r = by_key.get((m, d, cv_label))

            if r is None:
                # put "No data" in the center, not at the top
                ax.text(0.5, 0.55, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=PANEL_FS)
                ax.set_xticks([]); ax.set_yticks([])
                # title drawn *inside* the axes so it can’t touch anything above
                ax.set_title(f"{m.upper()}–{d.capitalize()}",
                             fontsize=PANEL_FS, y=0.88)   # was 0.97
                continue

            counts, _, _ = ax.hist(r["perm_R2"], bins='auto')
            ax.axvline(r["obs_R2"], linewidth=2)

            y_max = float(np.max(counts)) if counts.size else 1.0
            extra = max(2.0, 0.15 * y_max)
            ax.set_ylim(0, y_max + extra)

            xmin = float(min(np.min(r["perm_R2"]), r["obs_R2"]))
            xmax = float(max(np.max(r["perm_R2"]), r["obs_R2"]))
            span = xmax - xmin if xmax > xmin else 1.0
            ax.set_xlim(xmin - 0.08*span, xmax + 0.10*span)

            ax.text(0.62, 0.85,  # a bit lower to leave headroom for the title
                    f"obs R² = {r['obs_R2']:.3f}\np = {r['p_value']:.4f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=ANNOT_FS-1,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))

            if i == len(models) - 1:
                ax.set_xlabel("R² under permutation", fontsize=PANEL_FS)
            if j == 0:
                ax.set_ylabel("Count", fontsize=PANEL_FS)

            # title inside the axes to avoid any collision
            ax.set_title(f"{m.upper()}–{d.capitalize()}",
                         fontsize=PANEL_FS, y=1.02)

            ax.tick_params(labelsize=TICK_FS)


    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.06)
    plt.savefig(outfile, dpi=600, pad_inches=0.2)
    plt.close(fig)
    print(f" Saved matrix: {outfile}")




def _plot_pvalue_heatmap(by_key, models, descs, cv_label, outfile, vmin=0.0, vmax=None):
    pv = pd.DataFrame(index=models, columns=descs, dtype=float)
    for m in models:
        for d in descs:
            rec = by_key.get((m, d, cv_label))
            pv.loc[m, d] = rec["p_value"] if rec else np.nan

    data = -np.log10(pv.astype(float).clip(lower=1e-12))
    if vmax is None:
        vmax = np.nanmax(data.to_numpy())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(descs)));  ax.set_xticklabels([c.capitalize() for c in descs])
    ax.set_yticks(range(len(models))); ax.set_yticklabels([m.upper() for m in models])
    for i in range(len(models)):
        for j in range(len(descs)):
            val = pv.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("-log10(p)")
    ax.set_title(f"Y-scrambling p-values ({cv_label})")
    plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close(fig)
    print(f" Saved heatmap: {outfile}")


def build_scrambling_matrices(scramble_root: str,
                              models=MODELS, make_cv_heatmaps=True, descs=DESCS):
    os.makedirs(scramble_root, exist_ok=True)
    recs = _collect_records(scramble_root)
    by_key = {(r["model"], r["desc"], r["cv"]): r for r in recs}

    #  NEW BLOCK: save one standalone histogram per record
    for r in recs:
        stem = os.path.splitext(os.path.basename(r["path"]))[0]
        out = os.path.join(scramble_root, f"{stem}.single.png")
        _save_histogram(
            r["perm_R2"],
            r["obs_R2"],
            f"{r['model'].upper()}–{r['desc'].capitalize()} {r['cv']}",
            out,
            p_val=r["p_value"],
            column="single"   # or "double" if want wider
        )

    # Existing combined matrices
    _plot_matrix(by_key, models, descs, "10-fold",
                 os.path.join(scramble_root, "r2_scrambling_matrix_10fold.png"))
    _plot_matrix(by_key, models, descs, "LOSO",
                 os.path.join(scramble_root, "r2_scrambling_matrix_loso.png"))

    if make_cv_heatmaps:
        _plot_pvalue_heatmap(by_key, models, descs, "10-fold",
            os.path.join(scramble_root, "r2_scrambling_pvalue_heatmap_10fold.png"))
        _plot_pvalue_heatmap(by_key, models, descs, "LOSO",
            os.path.join(scramble_root, "r2_scrambling_pvalue_heatmap_LOSO.png"))

        
def _save_histogram(perm_R2, obs_R2, title, out_png,
                    p_val=None,
                    column="single",   # "single" ≈ 3.3×2.5 in, "double" ≈ 7.0×4.5 in
                    font=9):
    # layout
    figsize = (3.3, 2.5) if column == "single" else (7.0, 4.5)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # histogram + obs line
    counts, _, _ = ax.hist(perm_R2, bins='auto', zorder=1)
    ax.axvline(obs_R2, linewidth=2, zorder=2)

    # x-lims with a bit of extra right padding (avoid hugging edge)
    xmin = float(min(np.min(perm_R2), obs_R2))
    xmax = float(max(np.max(perm_R2), obs_R2))
    span = xmax - xmin if xmax > xmin else 1.0
    ax.set_xlim(xmin - 0.08 * span, xmax + 0.10 * span)

    # y headroom: fractional + absolute padding so top tick never clips
    y_max = float(np.max(counts)) if counts.size else 1.0
    extra = max(2.0, 0.15 * y_max)
    ax.set_ylim(0, y_max + extra)

    # labels & ticks
    ax.set_title(title, fontsize=font + 1, pad=3)
    ax.set_xlabel("R² under permutation", fontsize=font)
    ax.set_ylabel("Count", fontsize=font)
    ax.tick_params(labelsize=font)

    # centered annotation (with optional p-value), slightly lower than top
    text = f"obs R² = {obs_R2:.3f}"
    if p_val is not None:
        text += f"\np = {p_val:.4f}"
    ax.text(0.5, 0.88, text,
            transform=ax.transAxes, ha="center", va="top", fontsize=font,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))

    # save (avoid bbox_inches='tight' to prevent trimming ticks)
    plt.savefig(out_png, dpi=600, pad_inches=0.2)
    plt.close(fig)



if __name__ == "__main__":
    import argparse

    base_dir = os.path.dirname(__file__)
    default_root = os.path.join(base_dir, "outputs", "scrambling")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=default_root,
        help="Folder (searched recursively) for *_r2_scrambling.csv"
    )
    ap.add_argument(
        "--no-heatmaps", action="store_true",
        help="Skip p-value heatmaps"
    )
    ap.add_argument(
        "--debug", action="store_true",
        help="Print found keys/files"
    )
    args = ap.parse_args()

    build_scrambling_matrices(
        args.root,
        make_cv_heatmaps=not args.no_heatmaps
    )

