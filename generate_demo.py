"""
generate_demo.py — fabrique un jeu de données synthétique cohérent avec
l'encodeur factice (MockEmbedder), pour démontrer le pipeline de bout en bout.

Les "images" sont décrites par des labels multi-mots ; l'encodeur factice les
rend proches lorsqu'elles partagent des mots. Une fraction des mentations
reprend des mots de la cible (signal) ; le reste est du bruit. On obtient ainsi
un taux de succès au-dessus du hasard, suffisant pour exercer toute l'analyse.

Usage :
    python generate_demo.py            # écrit ./demo_data/
    python psicx.py run --data demo_data --config demo_data/config.json \
        --out demo_out --embedder mock
"""
from __future__ import annotations
import csv
import json
import random
from pathlib import Path

# Pool d'images : (id, label). Regroupées par familles sémantiques implicites.
POOL = [
    ("img_tour_eiffel",   "tour eiffel monument paris"),
    ("img_pont_golden",   "pont golden gate monument rouge"),
    ("img_phare_breton",  "phare breton tour cotiere"),
    ("img_arc_triomphe",  "arc triomphe monument paris"),
    ("img_chat_noir",     "chat noir animal domestique"),
    ("img_chien_berger",  "chien berger animal domestique"),
    ("img_cheval_blanc",  "cheval blanc animal ferme"),
    ("img_aigle_royal",   "aigle royal animal oiseau"),
    ("img_voiture_rouge", "voiture rouge vehicule route"),
    ("img_velo_ville",    "velo ville vehicule route"),
    ("img_train_vapeur",  "train vapeur vehicule rail"),
    ("img_bateau_voile",  "bateau voile vehicule mer"),
    ("img_pomme_verte",   "pomme verte fruit nourriture"),
    ("img_pain_campagne", "pain campagne nourriture"),
    ("img_cafe_tasse",    "cafe tasse boisson nourriture"),
    ("img_sculpture_abs", "sculpture abstraite art moderne"),
    ("img_tableau_cubiste","tableau cubiste art moderne"),
    ("img_fractale_bleue","fractale bleue motif abstrait"),
    ("img_montagne_neige","montagne neige paysage nature"),
    ("img_plage_sable",   "plage sable paysage mer"),
    ("img_foret_pins",    "foret pins paysage nature"),
    ("img_marteau_bois",  "marteau bois outil atelier"),
    ("img_cle_metal",     "cle metal outil porte"),
    ("img_lampe_bureau",  "lampe bureau objet maison"),
]

ANCHORS = ["animal", "monument", "vehicule", "nourriture", "paysage",
           "outil", "objet", "art", "personne", "batiment"]

SUBJECTS = [
    ("S01", ["monument", "tour", "pont", "ville", "paris"]),      # familier des repères urbains
    ("S02", ["animal", "chat", "chien", "ferme", "oiseau"]),      # familier des animaux
    ("S03", ["vehicule", "route", "mer", "rail"]),                # familier des transports
    ("S04", ["art", "moderne", "abstrait", "motif"]),             # familier de l'art
]


def build(seed: int = 7, n_trials: int = 80, signal_fraction: float = 0.45):
    rng = random.Random(seed)
    ids = [pid for pid, _ in POOL]
    labels = dict(POOL)

    # --- Ensembles de jugement : 1 cible + 3 leurres, position tirée au hasard.
    # On varie la difficulté : ~moitié des sets ont un leurre de la même famille
    # (mots partagés -> confusable), l'autre moitié des leurres dissemblables.
    sets = []
    for i in range(40):
        target = rng.choice(ids)
        fam_word = labels[target].split()[-1]  # mot de "famille" (dernier token)
        same_family = [x for x in ids if x != target and labels[x].split()[-1] == fam_word]
        others = [x for x in ids if x != target]
        if i % 2 == 0 and len(same_family) >= 1:        # set difficile
            d1 = rng.choice(same_family)
            rest = [x for x in others if x != d1]
            decoys = [d1] + rng.sample(rest, 2)
        else:                                            # set facile
            decoys = rng.sample(others, 3)
        rng.shuffle(decoys)
        sets.append({"set_id": f"set_{i:02d}", "target_id": target,
                     "decoy_1": decoys[0], "decoy_2": decoys[1], "decoy_3": decoys[2]})

    # --- Essais : sujet + set + mentation (signal ou bruit).
    noise_words = ["forme", "couleur", "lumiere", "ombre", "vague", "ligne",
                   "surface", "contour", "texture", "fond"]
    trials = []
    for j in range(n_trials):
        subj = rng.choice([s[0] for s in SUBJECTS])
        st = rng.choice(sets)
        tgt_words = labels[st["target_id"]].split()
        if rng.random() < signal_fraction:
            # mentation bruitée mais informative : 2 mots de la cible + bruit
            ment = rng.sample(tgt_words, k=min(2, len(tgt_words))) + rng.sample(noise_words, 2)
        else:
            ment = rng.sample(noise_words, 3)
        rng.shuffle(ment)
        trials.append({"trial_id": f"t{j:03d}", "subject_id": subj,
                       "set_id": st["set_id"], "mentation": " ".join(ment)})

    return labels, sets, trials


def write(out="demo_data"):
    labels, sets, trials = build()
    d = Path(out); d.mkdir(parents=True, exist_ok=True)

    with open(d / "targets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["target_id", "image_path", "label"])
        for pid, lab in POOL:
            w.writerow([pid, "", lab])

    with open(d / "sets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["set_id", "target_id", "decoy_1", "decoy_2", "decoy_3"])
        w.writeheader(); w.writerows(sets)

    with open(d / "subjects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["subject_id", "repertoire"])
        for sid, rep in SUBJECTS:
            w.writerow([sid, ";".join(rep)])

    with open(d / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trial_id", "subject_id", "set_id", "mentation"])
        w.writeheader(); w.writerows(trials)

    config = {
        "n_alternatives": 4,
        "difficulty_metric": "mean_decoy_sim",
        "complexity_metric": "prototypicality",
        "anchor_concepts": ANCHORS,
        "strata_quantiles": [1/3, 2/3],
        "repertoire_topk": 3,
        "weight_difficulty_exp": 1.0,
        "weight_conditional_exp": 1.0,
        "mc_iterations": 20000,
        "random_seed": 0,
    }
    (d / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Jeu de données synthétique écrit dans : {out}/")
    print(f"  {len(POOL)} images, {len(sets)} ensembles, {len(SUBJECTS)} sujets, {len(trials)} essais.")


if __name__ == "__main__":
    write()
