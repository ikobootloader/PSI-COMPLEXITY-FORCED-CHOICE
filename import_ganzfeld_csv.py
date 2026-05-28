"""
Import minimal d'un CSV Ganzfeld/forced-choice vers le format dataset psicx.

Usage:
  python import_ganzfeld_csv.py --in TransposedData.csv --out imported_data
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class MappedRow:
    trial_id: str
    subject_id: str
    set_id: str
    mentation: str
    target_id: str
    decoys: list[str]


TARGET_CANDIDATES = ("target_id", "target", "hit_target", "target_image", "target_label")
SUBJECT_CANDIDATES = ("subject_id", "subject", "participant", "receiver", "viewer")
TRIAL_CANDIDATES = ("trial_id", "trial", "id", "session_id")
SET_CANDIDATES = ("set_id", "set", "pool_id")
MENTATION_CANDIDATES = ("mentation", "transcript", "description", "notes", "report")
RANK_CANDIDATES = ("rank", "target_rank", "hit_rank", "position")

DECOY_HINTS = ("decoy", "distractor", "lure", "foil", "non_target", "option")


def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in s.strip().replace("-", "_").replace(" ", "_"))


def _first_existing(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    idx = {_norm(f): f for f in fieldnames}
    for cand in candidates:
        if cand in idx:
            return idx[cand]
    return None


def _guess_decoy_columns(fieldnames: list[str], target_col: str) -> list[str]:
    norm_map = {f: _norm(f) for f in fieldnames}
    cols: list[str] = []
    for f in fieldnames:
        nf = norm_map[f]
        if f == target_col:
            continue
        if any(h in nf for h in DECOY_HINTS):
            cols.append(f)
    return cols


def _build_mapped_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> list[MappedRow]:
    target_col = _first_existing(fieldnames, TARGET_CANDIDATES)
    if not target_col:
        raise ValueError("Impossible de trouver la colonne cible (target).")

    decoy_cols = _guess_decoy_columns(fieldnames, target_col)
    if len(decoy_cols) < 3:
        raise ValueError(
            "Impossible de trouver au moins 3 colonnes de leurres (decoy/lure/distractor/option)."
        )

    trial_col = _first_existing(fieldnames, TRIAL_CANDIDATES)
    subj_col = _first_existing(fieldnames, SUBJECT_CANDIDATES)
    set_col = _first_existing(fieldnames, SET_CANDIDATES)
    mentation_col = _first_existing(fieldnames, MENTATION_CANDIDATES)

    mapped: list[MappedRow] = []
    for i, row in enumerate(rows):
        target = (row.get(target_col) or "").strip()
        if not target:
            continue
        decoys = []
        for c in decoy_cols:
            v = (row.get(c) or "").strip()
            if v and v != target:
                decoys.append(v)
        # Dedup stable, puis tranche a 3 leurres car format psicx courant.
        seen = set()
        decoys = [x for x in decoys if not (x in seen or seen.add(x))]
        if len(decoys) < 3:
            continue
        decoys = decoys[:3]

        trial_id = ((row.get(trial_col) if trial_col else "") or f"t{i:04d}").strip() or f"t{i:04d}"
        subject_id = ((row.get(subj_col) if subj_col else "") or "S01").strip() or "S01"
        set_id = ((row.get(set_col) if set_col else "") or f"set_{i:04d}").strip() or f"set_{i:04d}"
        mentation = ((row.get(mentation_col) if mentation_col else "") or "").strip()

        mapped.append(
            MappedRow(
                trial_id=trial_id,
                subject_id=subject_id,
                set_id=set_id,
                mentation=mentation,
                target_id=target,
                decoys=decoys,
            )
        )
    if not mapped:
        raise ValueError("Aucune ligne exploitable apres mapping (cible + >=3 leurres).")
    return mapped


def import_csv(in_csv: Path, out_dir: Path) -> dict[str, int]:
    with open(in_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError("CSV sans en-tetes.")
        fieldnames = list(reader.fieldnames)

    mapped = _build_mapped_rows(rows, fieldnames)

    out_dir.mkdir(parents=True, exist_ok=True)

    target_ids = sorted({m.target_id for m in mapped} | {d for m in mapped for d in m.decoys})
    with open(out_dir / "targets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target_id", "image_path", "label"])
        for tid in target_ids:
            w.writerow([tid, "", tid])

    with open(out_dir / "sets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set_id", "target_id", "decoy_1", "decoy_2", "decoy_3"])
        for m in mapped:
            w.writerow([m.set_id, m.target_id, m.decoys[0], m.decoys[1], m.decoys[2]])

    subject_ids = sorted({m.subject_id for m in mapped})
    with open(out_dir / "subjects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "repertoire"])
        for sid in subject_ids:
            w.writerow([sid, ""])

    with open(out_dir / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "subject_id", "set_id", "mentation"])
        for m in mapped:
            w.writerow([m.trial_id, m.subject_id, m.set_id, m.mentation])

    cfg = {
        "n_alternatives": 4,
        "difficulty_metric": "mean_decoy_sim",
        "complexity_metric": "prototypicality",
        "anchor_concepts": [
            "animal",
            "monument",
            "vehicule",
            "nourriture",
            "paysage",
            "outil",
            "objet",
            "art",
            "personne",
            "batiment",
        ],
        "strata_quantiles": [1 / 3, 2 / 3],
        "repertoire_topk": 3,
        "weight_difficulty_exp": 1.0,
        "weight_conditional_exp": 0.0,
        "mc_iterations": 5000,
        "random_seed": 0,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "n_trials": len(mapped),
        "n_subjects": len(subject_ids),
        "n_targets": len(target_ids),
    }


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X~Binomial(n,p), exact."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    s = 0.0
    for x in range(k, n + 1):
        s += math.comb(n, x) * (p**x) * ((1.0 - p) ** (n - x))
    return min(max(s, 0.0), 1.0)


def rank_only_analysis(in_csv: Path, out_dir: Path, n_alternatives: int) -> dict[str, float]:
    with open(in_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError("CSV sans en-tetes.")
        fieldnames = list(reader.fieldnames)

    rank_col = _first_existing(fieldnames, RANK_CANDIDATES)
    if not rank_col:
        raise ValueError("Mode rank-only: colonne rank introuvable.")

    trial_col = _first_existing(fieldnames, TRIAL_CANDIDATES)
    subj_col = _first_existing(fieldnames, SUBJECT_CANDIDATES)
    target_col = _first_existing(fieldnames, TARGET_CANDIDATES)

    kept = []
    for i, row in enumerate(rows):
        rv = (row.get(rank_col) or "").strip()
        if not rv:
            continue
        try:
            r = int(float(rv))
        except ValueError:
            continue
        if r < 1 or r > n_alternatives:
            continue
        kept.append(
            {
                "trial_id": ((row.get(trial_col) if trial_col else "") or f"t{i:04d}").strip() or f"t{i:04d}",
                "subject_id": ((row.get(subj_col) if subj_col else "") or "NA").strip() or "NA",
                "target_id": ((row.get(target_col) if target_col else "") or "").strip(),
                "target_rank": r,
                "hit": int(r == 1),
            }
        )

    n = len(kept)
    if n == 0:
        raise ValueError("Mode rank-only: aucune ligne exploitable.")
    hits = sum(x["hit"] for x in kept)
    mce = 1.0 / float(n_alternatives)
    pval = _binom_sf(hits, n, mce)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rank_only_trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trial_id", "subject_id", "target_id", "target_rank", "hit"])
        w.writeheader()
        w.writerows(kept)

    summary = {
        "mode": "rank-only",
        "n_trials": n,
        "hits": hits,
        "mce": mce,
        "raw_hit_rate": hits / n,
        "binomial_p_one_sided": pval,
        "n_alternatives": n_alternatives,
    }
    (out_dir / "rank_only_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Importe un CSV Ganzfeld vers un dataset psicx.")
    ap.add_argument("--in", dest="in_csv", required=True, help="CSV source (ex: Transposed Data.csv)")
    ap.add_argument("--out", dest="out_dir", required=True, help="Dossier dataset de sortie")
    ap.add_argument(
        "--mode",
        choices=("full", "rank-only"),
        default="full",
        help="full: conversion dataset psicx complet; rank-only: analyse minimale a partir d'une colonne de rang",
    )
    ap.add_argument(
        "--n-alternatives",
        type=int,
        default=4,
        help="Nombre d'alternatives pour le calcul du hasard en mode rank-only (defaut: 4).",
    )
    args = ap.parse_args()

    if args.mode == "rank-only":
        summary = rank_only_analysis(Path(args.in_csv), Path(args.out_dir), args.n_alternatives)
        print(f"Analyse rank-only terminee vers {args.out_dir}")
        print(
            f"  trials={summary['n_trials']} hits={summary['hits']} "
            f"hit_rate={summary['raw_hit_rate']:.6f} p={summary['binomial_p_one_sided']:.6g}"
        )
    else:
        stats = import_csv(Path(args.in_csv), Path(args.out_dir))
        print(f"Import termine vers {args.out_dir}")
        print(
            f"  trials={stats['n_trials']} subjects={stats['n_subjects']} targets={stats['n_targets']}"
        )


if __name__ == "__main__":
    main()
