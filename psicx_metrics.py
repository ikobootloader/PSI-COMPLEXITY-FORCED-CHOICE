"""
Métriques pour psicx : difficulté, complexité absolue et conditionnelle.

Les métriques quantifient :
- Difficulté intra-ensemble (similarité cible-leurres)
- Complexité absolue des cibles (prototypicalité ou entropie visuelle)
- Complexité conditionnelle (familiarité sujet-cible)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from psicx_embedders import Embedder
from psicx_models import Target, JudgingSet, Config
from psicx_io import validate_file_path


def embed_pool(embedder: Embedder, targets: Sequence[Target]) -> dict[str, np.ndarray]:
    """Vectorise le pool d'images. Utilise l'encodeur d'images si un fichier
    existe, sinon le label textuel (ou le nom de fichier)."""
    img_items, txt_items = [], []
    for t in targets:
        if t.image_path:
            try:
                validated_path = validate_file_path(t.image_path)
                if validated_path.exists():
                    img_items.append(t)
                else:
                    txt_items.append(t)
            except ValueError:
                # Chemin invalide, utiliser le fallback textuel
                txt_items.append(t)
        else:
            txt_items.append(t)
    vecs: dict[str, np.ndarray] = {}
    if img_items:
        M = embedder.embed_images([t.image_path for t in img_items])
        for t, v in zip(img_items, M):
            vecs[t.target_id] = v
    if txt_items:
        keys = []
        for t in txt_items:
            if t.label:
                keys.append(t.label)
            elif t.image_path:
                keys.append(Path(t.image_path).stem.replace("_", " "))
            else:
                keys.append(t.target_id)
        M = embedder.embed_texts(keys)
        for t, v in zip(txt_items, M):
            vecs[t.target_id] = v
    return vecs


def set_difficulty(V: dict[str, np.ndarray], s: JudgingSet, metric: str) -> float:
    """Difficulté intra-ensemble : plus les leurres ressemblent à la cible,
    plus le set est difficile (et plus un succès « vaut »)."""
    if s.target_id not in V:
        raise KeyError(f"Target {s.target_id!r} (set {s.set_id!r}) absent du pool d'embeddings")
    t = V[s.target_id]
    sims = []
    for d in s.decoy_ids:
        if d not in V:
            raise KeyError(f"Decoy {d!r} (set {s.set_id!r}) absent du pool d'embeddings")
        sims.append(max(0.0, float(t @ V[d])))
    if not sims:
        return 0.0
    if metric == "max_decoy_sim":
        return max(sims)
    return float(np.mean(sims))


def visual_entropy(image_path: str, bins: int = 256) -> float:
    """Entropie de Shannon de l'histogramme de niveaux de gris (proxy de
    complexité ABSOLUE, clin d'oeil au gradient d'entropie de May)."""
    try:
        from PIL import Image
    except Exception:
        return float("nan")
    try:
        validated_path = validate_file_path(image_path)
    except ValueError:
        return float("nan")
    img = Image.open(validated_path).convert("L")
    hist = np.asarray(img.histogram()[:bins], dtype=float)
    p = hist / hist.sum() if hist.sum() else hist
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def absolute_complexity(
    V: dict[str, np.ndarray],
    targets: Sequence[Target],
    cfg: Config,
    anchors: np.ndarray | None,
) -> dict[str, float]:
    """Complexité absolue par cible. 'prototypicality' : 1 - (proximité max
    aux concepts d'ancrage) ; haute typicalité => faible complexité."""
    out: dict[str, float] = {}
    if cfg.complexity_metric == "visual_entropy":
        for t in targets:
            out[t.target_id] = visual_entropy(t.image_path) if t.image_path and Path(t.image_path).exists() else float("nan")
        return out
    # prototypicality
    if anchors is None or len(anchors) == 0:
        # pas d'ancrages -> pas d'information ; complexité neutre
        for t in targets:
            out[t.target_id] = 0.5
        return out
    for t in targets:
        proto = float(np.max(anchors @ V[t.target_id]))
        out[t.target_id] = 1.0 - max(0.0, min(1.0, proto))
    return out


def stratify(scores: dict[str, float], quantiles: Sequence[float]) -> dict[str, str]:
    """Range les cibles en strates Simple / Moyen / Complexe par quantiles."""
    vals = np.array([v for v in scores.values() if not math.isnan(v)])
    if vals.size == 0:
        return {k: "n/a" for k in scores}
    qs = np.quantile(vals, quantiles)
    labels = ("Simple", "Moyen", "Complexe")
    out = {}
    for k, v in scores.items():
        if math.isnan(v):
            out[k] = "n/a"
        else:
            idx = int(np.searchsorted(qs, v, side="right"))
            out[k] = labels[min(idx, len(labels) - 1)]
    return out


def conditional_complexity(target_vec: np.ndarray, R: np.ndarray | None, topk: int) -> float:
    """Complexité conditionnelle d'une cible pour un sujet : 1 - familiarité,
    où la familiarité est la proximité moyenne (top-k) au répertoire déclaré
    du sujet. Hypothèse forte assumée : familiarité ~ proximité d'embedding."""
    if R is None or len(R) == 0:
        return float("nan")
    sims = R @ target_vec
    k = min(topk, len(sims))
    top = np.sort(sims)[::-1][:k]
    fam = float(np.clip(np.mean(top), 0.0, 1.0))
    return 1.0 - fam
