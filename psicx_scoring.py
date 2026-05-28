"""
Scoring et inférence statistique pour psicx.

Fournit :
- score_trial : jugement objectif par similarité d'embeddings
- binom_sf_ge : test binomial exact (stable en log-espace)
- montecarlo_weighted_null : distribution nulle du score pondéré
"""
from __future__ import annotations

import math

import numpy as np

from psicx_models import JudgingSet

# Constante pour comparaisons flottantes
EPSILON_COMPARISON = 1e-12


def score_trial(V: dict[str, np.ndarray], s: JudgingSet, mentation_vec: np.ndarray):
    """Jugement objectif : on classe les n candidats par similarité à la
    mentation. Hit si la cible arrive première."""
    cand = [s.target_id] + list(s.decoy_ids)
    # Vérification existence de tous les candidats
    for c in cand:
        if c not in V:
            raise KeyError(f"Candidat {c!r} (set {s.set_id!r}) absent du pool d'embeddings")
    C = np.stack([V[c] for c in cand])
    sims = C @ mentation_vec
    hit = 1 if int(np.argmax(sims)) == 0 else 0
    order = np.argsort(-sims)
    target_rank = int(np.where(order == 0)[0][0]) + 1
    margin = float(sims[0] - np.max(sims[1:])) if len(sims) > 1 else float(sims[0])
    return hit, target_rank, margin


def _logpmf_binom(i: int, n: int, lp: float, lq: float) -> float:
    """Log-PMF binomiale (stable numériquement)."""
    return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * lp + (n - i) * lq)


def binom_sf_ge(k: int, n: int, p: float) -> float:
    """P(X >= k) pour X ~ Binomiale(n, p), en log-espace (stable)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    lp, lq = math.log(p), math.log(1 - p)
    terms = [_logpmf_binom(i, n, lp, lq) for i in range(k, n + 1)]
    m = max(terms)
    return math.exp(m + math.log(sum(math.exp(t - m) for t in terms)))


def montecarlo_weighted_null(weights_norm: np.ndarray, observed: float,
                             mce: float, iterations: int, seed: int):
    """Distribution nulle du taux de succès PONDÉRÉ. Sous H0 (aucun signal,
    cible tirée au hasard parmi les n), chaque hit ~ Bernoulli(mce),
    indépendamment des poids (qui sont fixés)."""
    rng = np.random.default_rng(seed)
    n = len(weights_norm)
    H = (rng.random((iterations, n)) < mce).astype(float)
    stats = (H * weights_norm).mean(axis=1)
    p = float((stats >= observed - EPSILON_COMPARISON).mean())
    return p, float(stats.mean()), float(stats.std())
