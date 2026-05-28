"""
Tests des invariants de psicx.  Lancer :  python tests_psicx.py
(Aucune dépendance de test externe : assertions simples.)
"""
import math
import numpy as np

# Import depuis le module principal qui réexporte tout
import psicx as P


def test_embedder_normalized_and_semantic():
    e = P.MockEmbedder(seed=1)
    V = e.embed_texts(["tour eiffel paris", "tour eiffel monument", "chat noir animal"])
    norms = np.linalg.norm(V, axis=1)
    assert np.allclose(norms, 1.0), "vecteurs non normalisés"
    sim_shared = float(V[0] @ V[1])      # partagent 'tour' 'eiffel'
    sim_disjoint = float(V[0] @ V[2])    # aucun mot commun
    assert sim_shared > sim_disjoint, "les mots partagés devraient rapprocher"


def test_binom_sf():
    assert math.isclose(P.binom_sf_ge(0, 10, 0.25), 1.0)
    assert P.binom_sf_ge(11, 10, 0.25) == 0.0
    # monotonie décroissante en k
    vals = [P.binom_sf_ge(k, 40, 0.25) for k in range(0, 41)]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))
    # à k = espérance, p ~ 0.5 (un peu plus)
    assert 0.3 < P.binom_sf_ge(10, 40, 0.25) < 0.7


def test_mc_null_mean_matches_mce():
    w = np.ones(200)
    p, mean, sd = P.montecarlo_weighted_null(w, 0.25, mce=0.25, iterations=20000, seed=0)
    assert abs(mean - 0.25) < 0.01, f"moyenne nulle {mean} != MCE"


def test_score_trial_hit_logic():
    e = P.MockEmbedder(seed=2)
    pool = ["cible tres specifique unique", "leurre alpha", "leurre beta", "leurre gamma"]
    V = {f"id{i}": v for i, v in enumerate(e.embed_texts(pool))}
    s = P.JudgingSet("s", "id0", ["id1", "id2", "id3"])
    # mentation identique au label de la cible -> doit toucher, rang 1
    m = e.embed_texts(["cible tres specifique unique"])[0]
    hit, rank, margin = P.score_trial(V, s, m)
    assert hit == 1 and rank == 1 and margin > 0


def test_weights_uniform_when_exponents_zero():
    # a=b=0 -> tous les poids = 1 -> taux pondéré == taux brut
    cfg = P.Config(weight_difficulty_exp=0.0, weight_conditional_exp=0.0,
                   anchor_concepts=["animal", "objet"], mc_iterations=2000)
    e = P.MockEmbedder(seed=3)
    targets = [P.Target("a", label="chat animal"), P.Target("b", label="table objet"),
               P.Target("c", label="chien animal"), P.Target("d", label="lampe objet")]
    sets = [P.JudgingSet("s0", "a", ["b", "c", "d"])]
    subjects = [P.Subject("S", ["animal", "chat"])]
    trials = [P.Trial("t0", "S", "s0", "chat animal"),
              P.Trial("t1", "S", "s0", "table objet")]
    res = P.run_study(targets, sets, subjects, trials, e, cfg)
    s = res["summary"]
    assert abs(s["raw_hit_rate"] - s["weighted_hit_rate"]) < 1e-9


def test_plan_hash_stable_and_sensitive():
    c1 = P.Config(anchor_concepts=["x", "y"])
    c2 = P.Config(anchor_concepts=["x", "y"])
    c3 = P.Config(anchor_concepts=["x", "y"], weight_conditional_exp=0.0)
    assert c1.plan_hash() == c2.plan_hash(), "hash instable pour configs identiques"
    assert c1.plan_hash() != c3.plan_hash(), "hash insensible à un changement de config"


def test_conditional_complexity_bounds():
    e = P.MockEmbedder(seed=4)
    t = e.embed_texts(["tour eiffel monument"])[0]
    R_known = e.embed_texts(["tour monument paris"])      # proche
    R_unknown = e.embed_texts(["sardine moteur algebre"])  # lointain
    cc_known = P.conditional_complexity(t, R_known, topk=1)
    cc_unknown = P.conditional_complexity(t, R_unknown, topk=1)
    assert 0.0 <= cc_known <= 1.0 and 0.0 <= cc_unknown <= 1.0
    assert cc_known < cc_unknown, "une cible familière devrait être moins complexe"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as ex:
            failed += 1
            print(f"  FAIL  {t.__name__} :: {ex}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests réussis.")
    raise SystemExit(1 if failed else 0)
