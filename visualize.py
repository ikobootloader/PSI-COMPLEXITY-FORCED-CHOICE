"""
visualize.py — diagnostic graphique confondeur vs modérateur.

Lit le `per_trial.csv` produit par psicx et trace le taux de succès en fonction
de la difficulté du set et de la complexité conditionnelle. C'est le graphique
qui permet de trancher empiriquement :

  * Taux PLAT au niveau du hasard (MCE) sur tout l'axe  -> aucun signal ;
    la pondération n'apporte rien.
  * Taux AU-DESSUS du hasard qui DÉCROÎT quand la complexité conditionnelle
    augmente  -> signature d'un MODÉRATEUR (signal réel modulé par la
    familiarité de la cible pour le sujet).
  * Taux au-dessus du hasard mais INDÉPENDANT de la complexité conditionnelle
    -> l'effet n'est pas modulé par cette variable (lecture "contrôle objectif"
    suffisante ; b = 0).

Aucune dépendance lourde : numpy + matplotlib. La régression logistique est
ajustée à la main (IRLS).

Usage :
    python visualize.py --trials demo_out/per_trial.csv --out demo_out/figures
    # MCE lue dans summary.json voisin si présent, sinon --mce (défaut 0.25)
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette sobre, lisible (compatible daltonisme)
C_POINT = "#1b4965"
C_LINE = "#c1121f"
C_MCE = "#6c757d"
C_BAR = "#5fa8d3"


# ----------------------------------------------------------------------------
# Statistiques (numpy pur)
# ----------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96):
    """Intervalle de Wilson pour une proportion (meilleur que la normale en
    petits effectifs)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def logistic_fit(x: np.ndarray, y: np.ndarray, iters: int = 100):
    """Régression logistique à un prédicteur par IRLS. Renvoie (b0, b1) ou
    None si séparation / non-convergence."""
    if len(np.unique(y)) < 2:
        return None
    X = np.column_stack([np.ones_like(x, dtype=float), x.astype(float)])
    beta = np.zeros(2)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-6, None)
        z = eta + (y - p) / W
        XtW = X.T * W
        try:
            beta_new = np.linalg.solve(XtW @ X + 1e-8 * np.eye(2), XtW @ z)
        except np.linalg.LinAlgError:
            return None
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            return beta_new
        beta = beta_new
    return beta


def quantile_bins(values: np.ndarray, n_bins: int):
    """Bornes de bins par quantiles (effectifs ~égaux). Renvoie les indices de
    bin (0..n_bins-1) et les centres."""
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    qs[0] -= 1e-9
    idx = np.clip(np.searchsorted(qs, values, side="right") - 1, 0, n_bins - 1)
    centers = []
    for b in range(n_bins):
        sel = values[idx == b]
        centers.append(float(sel.mean()) if sel.size else float("nan"))
    return idx, np.array(centers)


# ----------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------
def load_trials(path: Path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    hit = np.array([int(r["hit"]) for r in rows])
    diff = np.array([float(r["set_difficulty"]) for r in rows])
    cc_raw = [r.get("conditional_complexity") for r in rows]
    cc = np.array([float(v) if v not in (None, "", "None") else np.nan for v in cc_raw])
    stratum = [r.get("target_stratum", "n/a") for r in rows]
    return hit, diff, cc, stratum


def find_mce(trials_path: Path, fallback: float):
    summ = trials_path.parent / "summary.json"
    if summ.exists():
        try:
            return float(json.load(open(summ))["mce"])
        except Exception:
            pass
    return fallback


# ----------------------------------------------------------------------------
# Tracés élémentaires
# ----------------------------------------------------------------------------
def _rate_vs_predictor(ax, x, hit, mce, n_bins, xlabel, title):
    """Points binnés (taux + IC Wilson) + tendance logistique + ligne MCE."""
    mask = ~np.isnan(x)
    x, hit = x[mask], hit[mask]
    if x.size == 0:
        ax.set_title(title + " (pas de données)")
        return
    idx, centers = quantile_bins(x, n_bins)
    rates, los, his = [], [], []
    for b in range(n_bins):
        sel = hit[idx == b]
        r, lo, hi = wilson_ci(int(sel.sum()), int(sel.size))
        rates.append(r); los.append(r - lo); his.append(hi - r)
    ax.errorbar(centers, rates, yerr=[los, his], fmt="o", color=C_POINT,
                capsize=3, markersize=6, label="taux par bin (IC 95 % Wilson)",
                zorder=3)
    beta = logistic_fit(x, hit)
    if beta is not None:
        gx = np.linspace(x.min(), x.max(), 100)
        gy = 1.0 / (1.0 + np.exp(-(beta[0] + beta[1] * gx)))
        ax.plot(gx, gy, color=C_LINE, lw=2, label="tendance logistique", zorder=2)
    ax.axhline(mce, color=C_MCE, ls="--", lw=1.5, label=f"hasard (MCE = {mce:g})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("taux de succès")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)


def _rate_by_stratum(ax, hit, stratum, mce):
    order = ["Simple", "Moyen", "Complexe"]
    rates, los, his, labels = [], [], [], []
    for lab in order:
        sel = hit[np.array([s == lab for s in stratum])]
        if sel.size == 0:
            continue
        r, lo, hi = wilson_ci(int(sel.sum()), int(sel.size))
        rates.append(r); los.append(r - lo); his.append(hi - r)
        labels.append(f"{lab}\n(n={sel.size})")
    xpos = np.arange(len(labels))
    ax.bar(xpos, rates, color=C_BAR, edgecolor=C_POINT, zorder=2,
           yerr=[los, his], capsize=4, error_kw={"ecolor": C_POINT})
    ax.axhline(mce, color=C_MCE, ls="--", lw=1.5, label=f"hasard (MCE = {mce:g})")
    ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_ylabel("taux de succès"); ax.set_ylim(0, 1)
    ax.set_title("Taux par strate de complexité absolue (cible)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")


def _heatmap(ax, diff, cc, hit, n_bins=3):
    mask = ~np.isnan(cc)
    d, c, h = diff[mask], cc[mask], hit[mask]
    if d.size == 0:
        ax.set_title("Interaction (pas de données)")
        return
    di, _ = quantile_bins(d, n_bins)
    ci, _ = quantile_bins(c, n_bins)
    grid = np.full((n_bins, n_bins), np.nan)
    counts = np.zeros((n_bins, n_bins), dtype=int)
    for a in range(n_bins):
        for b in range(n_bins):
            sel = h[(di == a) & (ci == b)]
            counts[a, b] = sel.size
            if sel.size:
                grid[a, b] = sel.mean()
    im = ax.imshow(grid, origin="lower", cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    for a in range(n_bins):
        for b in range(n_bins):
            if counts[a, b]:
                ax.text(b, a, f"{grid[a, b]:.2f}\nn={counts[a, b]}",
                        ha="center", va="center", fontsize=8)
    ax.set_xlabel("complexité conditionnelle →")
    ax.set_ylabel("difficulté du set →")
    ax.set_xticks(range(n_bins)); ax.set_yticks(range(n_bins))
    ax.set_xticklabels([f"q{b+1}" for b in range(n_bins)])
    ax.set_yticklabels([f"q{a+1}" for a in range(n_bins)])
    ax.set_title("Taux de succès : difficulté × complexité conditionnelle")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="taux de succès")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def make_figures(trials_path, out_dir, n_bins=4, mce=0.25):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    hit, diff, cc, stratum = load_trials(Path(trials_path))
    mce = find_mce(Path(trials_path), mce)
    paths = []

    # Figure phare : taux vs complexité conditionnelle (le diagnostic décisif)
    fig, ax = plt.subplots(figsize=(7, 5))
    _rate_vs_predictor(ax, cc, hit, mce, n_bins,
                       "complexité conditionnelle (familier → inconnu pour le sujet)",
                       "Diagnostic décisif — succès vs complexité conditionnelle")
    fig.tight_layout()
    p = out / "diagnostic_complexite_conditionnelle.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # Taux vs difficulté
    fig, ax = plt.subplots(figsize=(7, 5))
    _rate_vs_predictor(ax, diff, hit, mce, n_bins,
                       "difficulté du set (leurres ressemblants →)",
                       "Succès vs difficulté intra-ensemble")
    fig.tight_layout()
    p = out / "succes_vs_difficulte.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # Panneau récapitulatif 2×2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _rate_vs_predictor(axes[0, 0], diff, hit, mce, n_bins,
                       "difficulté du set →", "A · Succès vs difficulté")
    _rate_vs_predictor(axes[0, 1], cc, hit, mce, n_bins,
                       "complexité conditionnelle →", "B · Succès vs complexité conditionnelle")
    _rate_by_stratum(axes[1, 0], hit, stratum, mce)
    _heatmap(axes[1, 1], diff, cc, hit, n_bins=3)
    fig.suptitle("psicx — diagnostic confondeur vs modérateur", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out / "panneau_diagnostic.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(prog="visualize",
                                 description="Diagnostic graphique confondeur vs modérateur.")
    ap.add_argument("--trials", required=True, help="Chemin vers per_trial.csv")
    ap.add_argument("--out", required=True, help="Dossier de sortie des figures")
    ap.add_argument("--bins", type=int, default=4, help="Nombre de bins par quantiles")
    ap.add_argument("--mce", type=float, default=0.25, help="Espérance par hasard (si pas de summary.json)")
    args = ap.parse_args(argv)
    paths = make_figures(args.trials, args.out, n_bins=args.bins, mce=args.mce)
    print("Figures écrites :")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
