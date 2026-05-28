"""
Pipeline principal de psicx.

Orchestre l'analyse complète : calcul des métriques, scoring, pondération,
inférence statistique, et génération du rapport.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict

import numpy as np

from psicx_embedders import Embedder
from psicx_models import Config
from psicx_metrics import (
    embed_pool, set_difficulty, absolute_complexity,
    stratify, conditional_complexity
)
from psicx_scoring import score_trial, binom_sf_ge, montecarlo_weighted_null

# Constante pour poids
EPSILON_WEIGHT = 1e-6

__version__ = "0.1.0"


def run_study(targets, sets, subjects, trials, embedder: Embedder, cfg: Config) -> dict:
    """Exécute l'analyse complète d'une étude.

    Args:
        targets: Liste de Target
        sets: Liste de JudgingSet
        subjects: Liste de Subject
        trials: Liste de Trial
        embedder: Backend d'embeddings
        cfg: Configuration de l'analyse

    Returns:
        Dictionnaire avec clés 'summary', 'per_trial', 'targets'
    """
    sets_by_id = {s.set_id: s for s in sets}
    subj_by_id = {s.subject_id: s for s in subjects}

    # --- Vectorisation ---
    V = embed_pool(embedder, targets)
    anchors = embedder.embed_texts(cfg.anchor_concepts) if cfg.anchor_concepts else None
    R_by_subject = {
        s.subject_id: (embedder.embed_texts(s.repertoire) if s.repertoire else np.zeros((0, embedder.dim)))
        for s in subjects
    }

    # --- Étapes 1 & 2 : difficulté des sets, complexité & strates des cibles ---
    difficulty = {s.set_id: set_difficulty(V, s, cfg.difficulty_metric) for s in sets}
    complexity = absolute_complexity(V, targets, cfg, anchors)
    strata = stratify(complexity, cfg.strata_quantiles)

    # --- Étapes 3 & 4 : scoring des essais + complexité conditionnelle ---
    rows = []
    mentation_vecs = embedder.embed_texts([t.mentation for t in trials])
    for tr, mvec in zip(trials, mentation_vecs):
        s = sets_by_id[tr.set_id]
        hit, rank, margin = score_trial(V, s, mvec)
        R = R_by_subject.get(tr.subject_id)
        cc = conditional_complexity(V[s.target_id], R, cfg.repertoire_topk)
        rows.append({
            "trial_id": tr.trial_id, "subject_id": tr.subject_id, "set_id": tr.set_id,
            "target_id": s.target_id, "hit": hit, "target_rank": rank,
            "evidence_margin": round(margin, 6),
            "set_difficulty": round(difficulty[tr.set_id], 6),
            "conditional_complexity": (round(cc, 6) if not math.isnan(cc) else None),
            "target_stratum": strata.get(s.target_id, "n/a"),
        })

    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    mce = cfg.resolved_mce()
    raw_rate = hits / n if n else float("nan")
    binom_p = binom_sf_ge(hits, n, mce) if n else float("nan")

    # --- Poids (confondeur^a * modérateur^b), normalisés à moyenne 1 ---
    a, b = cfg.weight_difficulty_exp, cfg.weight_conditional_exp
    raw_w = []
    for r in rows:
        d = r["set_difficulty"]
        c = r["conditional_complexity"]
        wc = (c + EPSILON_WEIGHT) ** b if (c is not None) else 1.0
        wd = (d + EPSILON_WEIGHT) ** a
        raw_w.append(wd * wc)
    raw_w = np.array(raw_w, dtype=float) if n else np.array([])
    if n and raw_w.sum() > 0:
        w_norm = raw_w * n / raw_w.sum()       # moyenne 1
    else:
        w_norm = np.ones(n)
    for r, w in zip(rows, w_norm):
        r["weight"] = round(float(w), 6)

    hit_arr = np.array([r["hit"] for r in rows], dtype=float) if n else np.array([])
    weighted_rate = float((w_norm * hit_arr).mean()) if n else float("nan")
    mc_p, mc_mean, mc_sd = (montecarlo_weighted_null(w_norm, weighted_rate, mce,
                                                     cfg.mc_iterations, cfg.random_seed)
                            if n else (float("nan"),) * 3)

    # --- Diagnostics : confondeur vs modérateur ---
    def rate_by(key, predicate):
        sel = [r for r in rows if predicate(r)]
        return (sum(x["hit"] for x in sel) / len(sel), len(sel)) if sel else (float("nan"), 0)

    by_stratum = {lab: rate_by("s", lambda r, L=lab: r["target_stratum"] == L)
                  for lab in ("Simple", "Moyen", "Complexe")}
    cc_vals = [r["conditional_complexity"] for r in rows if r["conditional_complexity"] is not None]
    cc_diag = {}
    if cc_vals:
        med = float(np.median(cc_vals))
        cc_diag = {
            "median": round(med, 6),
            "rate_low_cc":  rate_by("", lambda r: r["conditional_complexity"] is not None and r["conditional_complexity"] <= med),
            "rate_high_cc": rate_by("", lambda r: r["conditional_complexity"] is not None and r["conditional_complexity"] > med),
        }
    dmed = float(np.median([r["set_difficulty"] for r in rows])) if n else float("nan")
    diff_diag = {
        "median": round(dmed, 6) if n else None,
        "rate_easy": rate_by("", lambda r: r["set_difficulty"] <= dmed),
        "rate_hard": rate_by("", lambda r: r["set_difficulty"] > dmed),
    } if n else {}

    summary = {
        "n_trials": n,
        "hits": hits,
        "mce": round(mce, 6),
        "raw_hit_rate": round(raw_rate, 6) if n else None,
        "binomial_p_one_sided": binom_p,
        "weighted_hit_rate": round(weighted_rate, 6) if n else None,
        "weighted_mc_p_one_sided": mc_p,
        "weighted_mc_null_mean": round(mc_mean, 6) if n else None,
        "weighted_mc_null_sd": round(mc_sd, 6) if n else None,
        "weights": {"difficulty_exp_a": a, "conditional_exp_b": b},
        "diagnostics": {
            "hit_rate_by_target_stratum": by_stratum,
            "conditional_complexity": cc_diag,
            "set_difficulty": diff_diag,
        },
        "provenance": {
            "psicx_version": __version__,
            "analysis_plan_hash": cfg.plan_hash(),
            "embedder": type(embedder).__name__,
            "embedding_dim": getattr(embedder, "dim", None),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "config": {**asdict(cfg), "strata_quantiles": list(cfg.strata_quantiles)},
    }
    targets_out = [{"target_id": t.target_id,
                    "absolute_complexity": (round(complexity[t.target_id], 6)
                                            if not math.isnan(complexity[t.target_id]) else None),
                    "stratum": strata.get(t.target_id, "n/a")} for t in targets]
    return {"summary": summary, "per_trial": rows, "targets": targets_out}


def print_report(results: dict):
    """Affiche un rapport textuel formaté des résultats."""
    s = results["summary"]
    p = s["provenance"]
    def fmt(x): return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else x
    print("=" * 68)
    print("  RAPPORT psicx — scoring par complexité conditionnelle")
    print("=" * 68)
    print(f"  Essais            : {s['n_trials']}   |  succès : {s['hits']}   |  MCE : {s['mce']}")
    print(f"  Taux brut         : {fmt(s['raw_hit_rate'])}   (p binomial unilatéral = {fmt(s['binomial_p_one_sided']):.4g})"
          if s['n_trials'] else "  (aucun essai)")
    print(f"  Taux pondéré      : {fmt(s['weighted_hit_rate'])}   (p Monte-Carlo = {fmt(s['weighted_mc_p_one_sided']):.4g})")
    print(f"      poids : difficulté^{s['weights']['difficulty_exp_a']} × complexité_cond^{s['weights']['conditional_exp_b']}")
    diag = s["diagnostics"]
    print("-" * 68)
    print("  Diagnostic — taux de succès par strate de complexité (cible) :")
    for lab, (rate, k) in diag["hit_rate_by_target_stratum"].items():
        print(f"      {lab:<9}: {fmt(rate)}  (n={k})")
    if diag["conditional_complexity"]:
        cc = diag["conditional_complexity"]
        lo, klo = cc["rate_low_cc"]; hi, khi = cc["rate_high_cc"]
        print("  Diagnostic — confondeur vs modérateur (complexité conditionnelle) :")
        print(f"      cc faible : {fmt(lo)} (n={klo})   |   cc élevée : {fmt(hi)} (n={khi})")
    print("-" * 68)
    print(f"  Plan d'analyse (hash) : {p['analysis_plan_hash'][:16]}…  | embedder : {p['embedder']}")
    print("=" * 68)
