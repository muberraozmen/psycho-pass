from __future__ import annotations

import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from openrouter import OpenRouter


__all__ = [
    "LexicalEncoder",
    "SemanticEncoder",
]


DEFAULT_LEXICAL_MAX_FEATURES = 4096
DEFAULT_LEXICAL_STOP_WORDS = "english"
DEFAULT_LEXICAL_MIN_DF = 1
DEFAULT_LEXICAL_MAX_DF = 0.95

DEFAULT_SEMANTIC_MODEL_NAME = "qwen/qwen3-embedding-8b"
DEFAULT_SEMANTIC_MAX_CONTEXT_TOKENS = 32000

API_KEY = os.environ["OPENROUTER_API_KEY"]


class LexicalEncoder():
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.vectorizer = TfidfVectorizer(
            max_features=cfg.get("max_features", DEFAULT_LEXICAL_MAX_FEATURES),
            stop_words=cfg.get("stop_words", DEFAULT_LEXICAL_STOP_WORDS),
            min_df=cfg.get("min_df", DEFAULT_LEXICAL_MIN_DF),
            max_df=cfg.get("max_df", DEFAULT_LEXICAL_MAX_DF),
        )

    def execute(self, text: list[str]) -> list[list[float]]:
        embeddings = np.array(
            self.vectorizer.fit_transform(text).toarray(), dtype=np.float32
        ).tolist()
        return embeddings


class SemanticEncoder():
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.model_name = cfg.get("model_name", DEFAULT_SEMANTIC_MODEL_NAME)
        self.max_context_tokens = cfg.get("max_context_tokens", DEFAULT_SEMANTIC_MAX_CONTEXT_TOKENS)
        self.batch_size = cfg.get("batch_size", 32)
        self.client = OpenRouter(api_key=API_KEY)

    def execute(self, text: list[str]) -> list[list[float]]:
        # import pdb; pdb.set_trace()
        try:
            # truncate text if max_context_tokens is set
            if self.max_context_tokens is not None:
                n = self.max_context_tokens
                inputs = [t[:n] if len(t) > n else t for t in text]
            else:
                inputs = text
            
            response = self.client.embeddings.generate(
                model=self.model_name,
                input=inputs,
            )
            embeddings = [[] for _ in range(len(text))]
            for r in response.data:
                idx = int(r.index)
                embeddings[idx] = r.embedding
            return embeddings

        except Exception as e:
            print(f"An error occurred: {e}")
            return []


class SemanticEncoder:
    def __init__(self, cfg: dict):
        # super().__init__() is only needed if inheriting from a base class
        self.cfg = cfg
        self.model_name = cfg.get("model_name", DEFAULT_SEMANTIC_MODEL_NAME)
        self.max_context_tokens = cfg.get("max_context_tokens", DEFAULT_SEMANTIC_MAX_CONTEXT_TOKENS)
        self.batch_size = cfg.get("batch_size", 32)
        self.client = OpenRouter(api_key=API_KEY)

    def execute(self, text: list[str]) -> list[list[float]]:
        if self.max_context_tokens is not None:
            n = self.max_context_tokens
            char_limit = n * 4  # rough heuristic (1 token ≈ 4 characters) 
            inputs = [t[:char_limit] if len(t) > char_limit else t for t in text]
        else:
            inputs = text
        
        all_embeddings = []

        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            
            response = self.client.embeddings.generate(
                model=self.model_name,
                input=batch,
            )
            
            batch_embeddings = [[] for _ in range(len(batch))]
            for r in response.data:
                idx = int(r.index)
                batch_embeddings[idx] = r.embedding
            
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


