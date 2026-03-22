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
        self.client = OpenRouter(api_key=API_KEY)

    def execute(self, text: list[str]) -> list[list[float]]:
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
            
            response_data = sorted(response.data, key=lambda r: int(r.index))
            embeddings = [r.embedding for r in response_data]
            
            return embeddings

        except Exception as e:
            print(f"An error occurred: {e}")
            return []
