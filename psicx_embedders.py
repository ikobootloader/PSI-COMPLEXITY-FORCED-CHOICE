"""
Backends d'embeddings pour psicx.

Fournit une interface commune (Embedder) et deux implémentations :
- MockEmbedder : encodeur déterministe pour tests/démo (SANS valeur sémantique)
- ClipEmbedder : backend multimodal réel via sentence-transformers
"""
from __future__ import annotations

import hashlib
from typing import Sequence
from pathlib import Path

import numpy as np


def _l2norm(X: np.ndarray) -> np.ndarray:
    """Normalise L2 : la similarité cosinus devient un simple produit scalaire."""
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


class Embedder:
    """Interface. Toutes les sorties sont L2-normalisées : la similarité
    cosinus se réduit alors à un produit scalaire."""
    dim: int

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_images(self, paths: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class MockEmbedder(Embedder):
    """Encodeur DÉTERMINISTE pour tests et démonstration du pipeline.

    AUCUNE valeur sémantique réelle — NE JAMAIS l'utiliser pour une analyse.
    Une chaîne est représentée par la moyenne des vecteurs (aléatoires mais
    déterministes) de ses mots ; deux chaînes partageant des mots sont donc
    plus proches. Cela suffit à exercer toute la logique du pipeline.
    """

    def __init__(self, dim: int = 128, seed: int = 0):
        self.dim = dim
        self.seed = seed
        self._wcache: dict[str, np.ndarray] = {}

    def _word_vec(self, w: str) -> np.ndarray:
        if w in self._wcache:
            return self._wcache[w]
        h = hashlib.sha256(f"{self.seed}:{w}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.standard_normal(self.dim)
        v = v / (np.linalg.norm(v) or 1.0)
        self._wcache[w] = v
        return v

    def _embed_one(self, s: str) -> np.ndarray:
        words = [w for w in str(s).lower().replace("_", " ").replace("-", " ").split() if w]
        if not words:
            words = [str(s).lower() or "vide"]
        return np.mean([self._word_vec(w) for w in words], axis=0)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return _l2norm(np.asarray([self._embed_one(t) for t in texts]))

    def embed_images(self, paths: Sequence[str]) -> np.ndarray:
        # En mode factice, on encode le nom de fichier comme s'il était un label.
        labels = [Path(p).stem.replace("_", " ") for p in paths]
        return self.embed_texts(labels)


class ClipEmbedder(Embedder):
    """Vrai backend multimodal via sentence-transformers (modèle CLIP).

    À exécuter dans un environnement où Hugging Face est accessible (le premier
    appel télécharge les poids). Texte et image partagent le même espace.
    """

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        from sentence_transformers import SentenceTransformer  # import paresseux
        from PIL import Image
        self._Image = Image
        self.model = SentenceTransformer(model_name)
        dim = self.model.get_sentence_embedding_dimension()
        # Certains modèles CLIP exposent None ici : on infère alors la dimension.
        if dim is None:
            probe = self.model.encode(["probe"], convert_to_numpy=True)
            dim = int(np.asarray(probe).shape[-1])
        self.dim = int(dim)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return _l2norm(np.asarray(self.model.encode(list(texts), convert_to_numpy=True)))

    def embed_images(self, paths: Sequence[str]) -> np.ndarray:
        imgs = [self._Image.open(p).convert("RGB") for p in paths]
        return _l2norm(np.asarray(self.model.encode(imgs, convert_to_numpy=True)))


def make_embedder(kind: str = "mock", model: str = "clip-ViT-B-32", **kw) -> Embedder:
    """Factory pour créer un embedder selon le type demandé."""
    if kind == "mock":
        return MockEmbedder(**kw)
    if kind == "clip":
        return ClipEmbedder(model_name=model)
    raise ValueError(f"Embedder inconnu : {kind!r} (attendu 'mock' ou 'clip').")
