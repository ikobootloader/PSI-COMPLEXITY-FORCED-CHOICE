# CHANGELOG

## [Non versionne] - Correctif CLIP

### Corrige
- `ClipEmbedder` : correction d'un crash quand `get_sentence_embedding_dimension()` retourne `None` sur certains modeles CLIP.
- Ajout d'un fallback robuste : infere la dimension via un encodage probe, puis initialise `dim`.
- Validation fonctionnelle : run `--embedder clip --model clip-ViT-B-32` execute avec succes.

## [Non versionné] - Refactoring et corrections

### Corrigé — Critique
- ✓ **Duplication structure** : Supprimé psi-complexity/ et psi-complexity.zip (fichiers identiques)
- ✓ **Validation données** : Ajout `validate_dataset()` avec vérifications :
  - Intégrité référentielle (target_id, decoy_ids dans targets.csv)
  - subject_id et set_id dans trials.csv existent
  - Cible non présente dans ses propres leurres
  - Exception `DataValidationError` avec messages explicites
- ✓ **Gestion erreurs KeyError** : Ajout vérifications explicites dans :
  - `score_trial()` : tous candidats vérifiés avant accès
  - `set_difficulty()` : target et decoys vérifiés avec messages d'erreur clairs
- ✓ **Sécurité chemins** : Ajout `validate_file_path()` :
  - Protection path traversal
  - Résolution sécurisée des chemins
  - Validation optionnelle dans base_dir
  - Utilisé dans `embed_pool()` et `visual_entropy()`

### Corrigé — Important
- ✓ **Architecture modulaire** : psicx.py (596→98 lignes) décomposé en :
  - [psicx_embedders.py](psicx_embedders.py) : Backends d'embeddings (MockEmbedder, ClipEmbedder)
  - [psicx_models.py](psicx_models.py) : Dataclasses (Target, JudgingSet, Subject, Trial, Config)
  - [psicx_metrics.py](psicx_metrics.py) : Métriques (difficulté, complexité absolue/conditionnelle)
  - [psicx_scoring.py](psicx_scoring.py) : Scoring et inférence statistique
  - [psicx_io.py](psicx_io.py) : I/O et validation
  - [psicx_pipeline.py](psicx_pipeline.py) : Orchestration (`run_study()`, `print_report()`)
  - [psicx.py](psicx.py) : CLI uniquement, réexporte tout pour compatibilité
- ✓ **Constantes magiques** : Extraction en constantes nommées :
  - `EPSILON_WEIGHT = 1e-6` (calcul poids)
  - `EPSILON_STABLE = 1e-8` (stabilité matrices)
  - `EPSILON_COMPARISON = 1e-12` (comparaisons flottantes)

### Tests
- ✓ 7/7 tests unitaires passent avec architecture modulaire
- ✓ Pipeline complet validé (generate_demo.py + CLI)
- ✓ Compatibilité arrière préservée (imports via psicx.py)

### Restant (non critique)
- Tests couverture erreurs (I/O, validation, visualize.py)
- Validation schéma JSON config
- Barre progression Monte-Carlo (20k itérations)

### Points forts maintenus
- Documentation hypothèses scientifiques
- Préenregistrement rigoureux (analysis_plan_hash)
- Backend embedder interchangeable
- Séparation confondeur/modérateur
