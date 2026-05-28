"""
psicx — Scoring par complexité conditionnelle pour expériences psi en choix forcé.
================================================================================

Implémente les quatre opérations décrites dans l'essai « Une variable de trop,
une variable oubliée » :

  1. Difficulté intra-ensemble : similarité sémantique entre la cible et ses
     leurres, pour rendre comparables des essais hétérogènes.
  2. Stratification des cibles par complexité (absolue).
  3. Profilage prudent du sujet -> complexité conditionnelle de chaque paire
     sujet-cible.
  4. Pondération des succès par la difficulté.

Le scoring du jugement (quelle image ressemble le plus à la mentation ?) est
RÉALISÉ PAR EMBEDDINGS, non par un juge humain. La probabilité de base reste
1/n (espérance par hasard) tant que la cible est tirée au hasard parmi les
n images de l'ensemble — ce que l'outil ne peut pas garantir à votre place.

------------------------------------------------------------------------------
AVERTISSEMENTS SCIENTIFIQUES (à lire avant tout usage réel)
------------------------------------------------------------------------------
* Les embeddings NE SONT PAS un sol neutre : ils encodent les mêmes prototypes
  culturels que l'on cherche à mesurer (biais datés, anglocentrés). Une mesure
  de proximité n'est pas une vérité cognitive.
* Le calcul de difficulté suppose toujours un ensemble candidat : la question de
  la sélection des leurres est déplacée, pas résolue.
* Le profilage et la pondération AJOUTENT des degrés de liberté analytiques.
  Sans préenregistrement intégral, ce dispositif peut fabriquer de la
  significativité. L'outil fige donc un hash de la configuration : committez la
  config AVANT de collecter les données.
* Confondeur vs modérateur : la pondération par difficulté (a) traite la
  variable comme un confondeur à neutraliser ; la pondération par complexité
  conditionnelle (b) suppose un signal réel qu'elle modulerait. Les deux rôles
  sont SÉPARÉS par les exposants `weight_difficulty_exp` (a) et
  `weight_conditional_exp` (b). Mettez b=0 pour rester dans la lecture
  purement "contrôle objectif".

Dépendance obligatoire : numpy. Optionnelles : sentence-transformers + Pillow
(backend CLIP), pour une analyse réelle.
"""
from __future__ import annotations

import argparse

# Imports depuis modules métier
from psicx_embedders import make_embedder
from psicx_io import load_dataset, load_config, write_outputs
from psicx_pipeline import run_study, print_report

# Réexport des classes principales pour compatibilité
from psicx_models import Target, JudgingSet, Subject, Trial, Config
from psicx_embedders import Embedder, MockEmbedder, ClipEmbedder
from psicx_metrics import (
    embed_pool, set_difficulty, visual_entropy,
    absolute_complexity, stratify, conditional_complexity
)
from psicx_scoring import score_trial, binom_sf_ge, montecarlo_weighted_null
from psicx_io import DataValidationError, validate_file_path, validate_dataset

__version__ = "0.1.0"


# ============================================================================
# CLI
# ============================================================================

def main(argv=None):
    """Point d'entrée CLI."""
    ap = argparse.ArgumentParser(
        prog="psicx",
        description="Scoring par complexité conditionnelle (expériences psi en choix forcé)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Analyser une étude.")
    r.add_argument("--data", required=True, help="Dossier contenant targets.csv, sets.csv, subjects.csv, trials.csv")
    r.add_argument("--config", required=True, help="Fichier de configuration JSON (le plan d'analyse).")
    r.add_argument("--out", required=True, help="Dossier de sortie.")
    r.add_argument("--embedder", default="mock", choices=["mock", "clip"])
    r.add_argument("--model", default="clip-ViT-B-32", help="Nom du modèle CLIP (backend clip).")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        cfg = load_config(args.config)
        targets, sets, subjects, trials = load_dataset(args.data)
        emb = make_embedder(args.embedder, model=args.model, seed=cfg.random_seed)
        if args.embedder == "mock":
            print("  [!] Backend MOCK : démonstration du pipeline uniquement, "
                  "aucune validité sémantique.\n")
        results = run_study(targets, sets, subjects, trials, emb, cfg)
        write_outputs(results, args.out)
        print_report(results)
        print(f"\n  Sorties écrites dans : {args.out}/")


if __name__ == "__main__":
    main()
