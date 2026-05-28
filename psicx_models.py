"""
Modèle de données pour psicx.

Définit les structures pour targets, sets, subjects, trials et configuration.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict


@dataclass
class Target:
    """Une image/cible du pool."""
    target_id: str
    image_path: str | None = None
    label: str | None = None  # description textuelle optionnelle


@dataclass
class JudgingSet:
    """Un ensemble de jugement : 1 cible + n leurres."""
    set_id: str
    target_id: str
    decoy_ids: list[str]


@dataclass
class Subject:
    """Un participant avec son répertoire conceptuel déclaré."""
    subject_id: str
    repertoire: list[str]  # concepts familiers, en texte


@dataclass
class Trial:
    """Un essai : sujet, ensemble, mentation produite."""
    trial_id: str
    subject_id: str
    set_id: str
    mentation: str


@dataclass
class Config:
    """Configuration complète de l'analyse (le plan d'analyse préenregistré)."""
    n_alternatives: int = 4
    difficulty_metric: str = "mean_decoy_sim"      # ou "max_decoy_sim"
    complexity_metric: str = "prototypicality"     # ou "visual_entropy"
    anchor_concepts: list = field(default_factory=list)   # pour prototypicality
    strata_quantiles: tuple = (1 / 3, 2 / 3)
    repertoire_topk: int = 3
    weight_difficulty_exp: float = 1.0             # a (rôle confondeur)
    weight_conditional_exp: float = 1.0            # b (rôle modérateur)
    mc_iterations: int = 20000
    mce: float | None = None                       # défaut = 1/n_alternatives
    random_seed: int = 0

    def resolved_mce(self) -> float:
        """MCE effective (1/n si non spécifiée)."""
        return float(self.mce) if self.mce is not None else 1.0 / self.n_alternatives

    def plan_hash(self) -> str:
        """SHA-256 de la configuration -> identifiant du plan d'analyse.
        À committer AVANT collecte des données (discipline de préenregistrement)."""
        d = asdict(self)
        d["strata_quantiles"] = list(self.strata_quantiles)
        blob = json.dumps(d, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
