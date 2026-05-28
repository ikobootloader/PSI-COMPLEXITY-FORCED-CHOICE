# psicx — scoring par complexité conditionnelle

Outil de recherche pour expériences psi en **choix forcé** (Ganzfeld, vision à
distance avec ensemble de jugement). Il implémente les quatre opérations de
l'essai *« Une variable de trop, une variable oubliée »* :

1. **Difficulté intra-ensemble** — similarité sémantique entre la cible et ses
   leurres, pour rendre comparables des essais hétérogènes.
2. **Stratification** des cibles par complexité absolue.
3. **Profilage** prudent du répertoire du sujet → complexité *conditionnelle*
   de chaque paire sujet-cible.
4. **Pondération** des succès par la difficulté.

Le jugement (« quelle image ressemble le plus à la mentation ? ») est fait par
**embeddings**, pas par un juge humain. La probabilité par hasard reste 1/n.

> ⚠️ Ce dépôt n'affirme rien sur l'existence du psi. C'est un instrument de
> mesure et de calibration du hasard, à utiliser **préenregistré**.

---

## Installation

```bash
# Obligatoire
pip install numpy

# Optionnel : backend d'embedding réel (CLIP) + lecture d'images
pip install sentence-transformers pillow
```

Aucune installation du paquet n'est nécessaire : `psicx.py` se lance directement.

## Démarrage rapide (démonstration hors-ligne)

```bash
python generate_demo.py                      # crée ./demo_data/ (données synthétiques)
python psicx.py run \
    --data demo_data \
    --config demo_data/config.json \
    --out demo_out \
    --embedder mock                          # encodeur factice : pipeline only
```

Le backend `mock` est **déterministe et sans valeur sémantique** : il sert
uniquement à exercer le pipeline. Pour une analyse réelle, utilisez `--embedder
clip` avec de vraies images.

## Analyse réelle

```bash
python psicx.py run --data mon_etude --config plan.json --out resultats \
    --embedder clip --model clip-ViT-B-32
```

Le premier appel CLIP télécharge les poids du modèle (nécessite un accès
réseau à Hugging Face).

---

## Formats d'entrée (dossier `--data`)

**`targets.csv`** — le pool d'images (toute image peut être cible ou leurre)
```
target_id,image_path,label
img_001,/chemin/vers/img_001.jpg,
img_002,/chemin/vers/img_002.jpg,
```
`label` est optionnel ; s'il manque et qu'aucune image n'est trouvée, le nom de
fichier sert de label (utile en mode `mock`).

**`sets.csv`** — les ensembles de jugement (1 cible + 3 leurres)
```
set_id,target_id,decoy_1,decoy_2,decoy_3
set_00,img_001,img_010,img_017,img_023
```
> La cible doit avoir été **tirée au hasard** parmi les n images du set *au
> moment de l'expérience* : c'est cette randomisation, et non l'outil, qui
> garantit l'espérance 1/n.

**`subjects.csv`** — répertoire conceptuel déclaré, concepts séparés par `;`
```
subject_id,repertoire
S01,monument;tour;pont;ville
```

**`trials.csv`** — les essais (la mentation est le texte produit par le sujet)
```
trial_id,subject_id,set_id,mentation
t000,S01,set_00,"je vois une grande structure verticale et de l'eau"
```

## Configuration (`config.json`) = le plan d'analyse

| Champ | Rôle |
|---|---|
| `n_alternatives` | nombre d'images par set (MCE = 1/n) |
| `difficulty_metric` | `mean_decoy_sim` ou `max_decoy_sim` |
| `complexity_metric` | `prototypicality` (avec `anchor_concepts`) ou `visual_entropy` (images requises) |
| `anchor_concepts` | concepts de référence pour la prototypicalité |
| `strata_quantiles` | bornes de strates, ex. `[0.333, 0.667]` |
| `repertoire_topk` | nb de concepts du répertoire agrégés pour la familiarité |
| `weight_difficulty_exp` (a) | exposant du **confondeur** (difficulté) |
| `weight_conditional_exp` (b) | exposant du **modérateur** (complexité cond.) |
| `mc_iterations` | itérations Monte-Carlo pour la nulle pondérée |
| `random_seed` | graine |

**Confondeur vs modérateur.** Mettez `b = 0` pour rester dans la lecture
« contrôle objectif » (la difficulté est neutralisée, sans supposer de signal).
Mettez `b > 0` pour la lecture « modérateur » (la complexité conditionnelle
module un effet supposé réel). Les deux ne doivent pas être confondues.

## Sorties (dossier `--out`)

- **`summary.json`** — taux brut + p binomial exact (test primaire), taux
  pondéré + p Monte-Carlo, diagnostics par strate et par niveau de complexité
  conditionnelle, et un bloc `provenance` contenant le **hash du plan
  d'analyse**.
- **`per_trial.csv`** — hit, rang de la cible, marge d'évidence, difficulté du
  set, complexité conditionnelle, poids — pour chaque essai.
- **`targets.csv`** — complexité absolue et strate de chaque cible.

### Lire les diagnostics
- *Taux par strate* : si les cibles « Simples » réussissent nettement mieux que
  les « Complexes », la difficulté pèse réellement sur les résultats.
- *cc faible vs élevée* : compare le taux de succès selon la complexité
  conditionnelle. Une différence soutient la lecture *modérateur* ; une absence
  de différence suggère que la pondération en `b` n'apporte rien.

---

## Préenregistrement (lire avant de collecter)

Le `analysis_plan_hash` est le SHA-256 de la configuration complète. La
discipline recommandée :

1. Figez `config.json` **avant** de collecter la moindre donnée.
2. Notez son hash (affiché en fin de rapport) dans votre préenregistrement.
3. Toute modification ultérieure change le hash : l'analyse n'est alors plus
   celle qui a été préenregistrée.

C'est la garde-fou contre le risque propre à cette méthode : le profilage et la
pondération ajoutent des degrés de liberté analytiques qui, non contraints,
peuvent fabriquer de la significativité.

## Limites assumées

- **Les embeddings ne sont pas neutres** : ils encodent les prototypes
  culturels (datés, anglocentrés) que l'on cherche à mesurer. Une proximité
  vectorielle n'est pas une vérité cognitive.
- **La sélection des leurres est déplacée, pas résolue** : tout calcul de
  difficulté suppose l'ensemble candidat.
- **Le profilage repose sur une hypothèse forte** : familiarité ≈ proximité
  d'embedding au répertoire *déclaré*. À documenter et discuter, jamais à
  présenter comme une mesure objective de la cognition du sujet.
- **MockEmbedder** ne vaut que pour tester le pipeline.

## Visualisation — le diagnostic décisif

```bash
pip install matplotlib
python visualize.py --trials demo_out/per_trial.csv --out demo_out/figures
# (la MCE est lue dans summary.json voisin si présent)
```

Produit trois PNG, dont `panneau_diagnostic.png` (4 vues) et la figure phare
`diagnostic_complexite_conditionnelle.png`. Chaque relation est tracée avec ses
intervalles de Wilson (95 %), une tendance logistique et la ligne de hasard.

**Comment lire la figure B (succès vs complexité conditionnelle)** — c'est elle
qui tranche entre les deux lectures de l'essai :

- Points **plats sur la ligne de hasard** → aucun signal ; la pondération en
  `b` est inutile.
- Points **au-dessus du hasard, en pente descendante** (succès plus élevé pour
  les cibles *familières* au sujet, à gauche) → signature d'un **modérateur** :
  un signal réel modulé par la familiarité. C'est ce qui justifierait `b > 0`.
- Points **au-dessus du hasard mais plats** → effet réel mais non modulé par
  cette variable ; la lecture « contrôle objectif » (`b = 0`) suffit.

La figure A (vs difficulté) et la heatmap d'interaction permettent de vérifier
que l'effet observé en B n'est pas un simple reflet de la difficulté des sets.

> Sur le jeu de démonstration (backend `mock`), les pentes n'ont **aucun sens** :
> elles ne servent qu'à montrer que les graphiques détectent et affichent une
> relation, dans un sens comme dans l'autre.

## Tests

```bash
python tests_psicx.py
```

## Fichiers

**Modules métier :**
- [psicx_embedders.py](psicx_embedders.py) — backends d'embeddings (mock/CLIP)
- [psicx_models.py](psicx_models.py) — structures de données
- [psicx_metrics.py](psicx_metrics.py) — difficulté, complexité absolue/conditionnelle
- [psicx_scoring.py](psicx_scoring.py) — scoring et inférence statistique
- [psicx_io.py](psicx_io.py) — I/O, validation, sécurité
- [psicx_pipeline.py](psicx_pipeline.py) — orchestration analyse complète

**CLI et utils :**
- [psicx.py](psicx.py) — CLI principal (réexporte modules pour compatibilité)
- [generate_demo.py](generate_demo.py) — jeu de données synthétique
- [visualize.py](visualize.py) — diagnostic graphique confondeur/modérateur
- [tests_psicx.py](tests_psicx.py) — tests unitaires (7 invariants)
- [requirements.txt](requirements.txt) — dépendances optionnelles
