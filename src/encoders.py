import os
from typing import Any
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from together import Together


__all__ = ["TFIDFEncoder", "TransformersEncoder", "TogetherEncoder"]


class BaseEncoder:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
    
    def run(self, text: list[str]) -> list[list[float]]:
        raise NotImplementedError("Subclasses must implement the run method.")


class TFIDFEncoder(BaseEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.vectorizer = TfidfVectorizer(
            max_features=cfg.get("max_features", 2000), 
            stop_words=cfg.get("stop_words", 'english'),
            min_df=cfg.get("min_df", 1),
            max_df=cfg.get("max_df", 0.95)
        )

    def run(self, text: list[str]) -> list[list[float]]:
        embeddings = np.array(self.vectorizer.fit_transform(text).toarray(), dtype=np.float32)
        return embeddings.tolist()


class TransformersEncoder(BaseEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = SentenceTransformer(cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        self.model.eval()

    def run(self, text: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(text)
        return embeddings.tolist()


class TogetherEncoder(BaseEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_name = cfg.get("model_name", "intfloat/multilingual-e5-large-instruct")
        self.client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    def run(self, text: list[str]) -> list[list[float]]:
        try:
            input = [t[:512] if len(t) > 512 else t for t in text]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=input
                )
            embeddings = [x.embedding for x in response.data]
            return embeddings

        except Exception as e:
            print(f"An error occurred: {e}")
            return []



    
