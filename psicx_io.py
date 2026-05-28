"""
Entrées/sorties et validation pour psicx.

Gère :
- Validation des chemins de fichiers (sécurité)
- Chargement des datasets (CSV)
- Validation de l'intégrité référentielle
- Chargement de la configuration
- Écriture des résultats
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from psicx_models import Target, JudgingSet, Subject, Trial, Config


class DataValidationError(Exception):
    """Erreur de validation de l'intégrité des données."""
    pass


def validate_file_path(path: str | Path, base_dir: Path | None = None) -> Path:
    """Valide et résout un chemin de fichier pour éviter path traversal.

    Args:
        path: Chemin à valider
        base_dir: Répertoire de base optionnel (si fourni, vérifie que le chemin résolu est dedans)

    Returns:
        Chemin résolu et validé

    Raises:
        ValueError: Si le chemin contient des éléments suspects ou sort du répertoire de base
    """
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Chemin invalide {path!r} : {e}")

    # Si un répertoire de base est spécifié, vérifier que le fichier est dedans
    if base_dir is not None:
        try:
            base_resolved = base_dir.resolve()
            resolved.relative_to(base_resolved)
        except (ValueError, OSError):
            raise ValueError(
                f"Chemin {path!r} sort du répertoire autorisé {base_dir}"
            )

    return resolved


def validate_dataset(targets, sets, subjects, trials):
    """Vérifie l'intégrité référentielle des données chargées.

    Lève DataValidationError si :
    - sets.csv référence des target_id ou decoy_ids inexistants
    - trials.csv référence des subject_id ou set_id inexistants
    - ensembles de jugement mal formés (nombre de leurres incorrect)
    """
    target_ids = {t.target_id for t in targets}
    set_ids = {s.set_id for s in sets}
    subject_ids = {s.subject_id for s in subjects}

    # Validation des sets
    for s in sets:
        if s.target_id not in target_ids:
            raise DataValidationError(
                f"Set {s.set_id!r} : target_id {s.target_id!r} inexistant dans targets.csv"
            )
        for i, decoy in enumerate(s.decoy_ids, 1):
            if decoy not in target_ids:
                raise DataValidationError(
                    f"Set {s.set_id!r} : decoy_{i} {decoy!r} inexistant dans targets.csv"
                )
        # Vérifier que la cible n'est pas dans ses propres leurres
        if s.target_id in s.decoy_ids:
            raise DataValidationError(
                f"Set {s.set_id!r} : la cible {s.target_id!r} apparaît parmi ses leurres"
            )

    # Validation des trials
    for tr in trials:
        if tr.subject_id not in subject_ids:
            raise DataValidationError(
                f"Trial {tr.trial_id!r} : subject_id {tr.subject_id!r} inexistant dans subjects.csv"
            )
        if tr.set_id not in set_ids:
            raise DataValidationError(
                f"Trial {tr.trial_id!r} : set_id {tr.set_id!r} inexistant dans sets.csv"
            )


def _read_csv(path: Path) -> list[dict]:
    """Lit un CSV et retourne une liste de dictionnaires."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_dataset(data_dir: str | Path):
    """Charge un dataset complet depuis un dossier.

    Args:
        data_dir: Dossier contenant targets.csv, sets.csv, subjects.csv, trials.csv

    Returns:
        (targets, sets, subjects, trials)

    Raises:
        DataValidationError: Si l'intégrité référentielle est violée
    """
    d = Path(data_dir)
    targets = [Target(r["target_id"], r.get("image_path") or None, r.get("label") or None)
               for r in _read_csv(d / "targets.csv")]
    sets = [JudgingSet(r["set_id"], r["target_id"],
                       [r[k] for k in ("decoy_1", "decoy_2", "decoy_3") if r.get(k)])
            for r in _read_csv(d / "sets.csv")]
    subjects = [Subject(r["subject_id"],
                        [x.strip() for x in (r.get("repertoire") or "").split(";") if x.strip()])
                for r in _read_csv(d / "subjects.csv")]
    trials = [Trial(r["trial_id"], r["subject_id"], r["set_id"], r["mentation"])
              for r in _read_csv(d / "trials.csv")]
    validate_dataset(targets, sets, subjects, trials)
    return targets, sets, subjects, trials


def load_config(path: str | Path) -> Config:
    """Charge une configuration depuis un fichier JSON."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = Config.__dataclass_fields__
    cfg = Config(**{k: v for k, v in d.items() if k in fields})
    if isinstance(cfg.strata_quantiles, list):
        cfg.strata_quantiles = tuple(cfg.strata_quantiles)
    return cfg


def write_outputs(results: dict, out_dir: str | Path):
    """Écrit les résultats d'une analyse dans un dossier."""
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / "summary.json").write_text(json.dumps(results["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
    for name, rows in (("per_trial.csv", results["per_trial"]), ("targets.csv", results["targets"])):
        if rows:
            with open(o / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
