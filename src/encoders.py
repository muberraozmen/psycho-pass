import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["TFIDFEncoder"]

# ------------------------------------------------------------
# Base Class
# ------------------------------------------------------------
class BaseEncoder:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def run(self, load_from: str, save_to: str) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement the run method.")

# ------------------------------------------------------------
# TFIDF Encoder
# ------------------------------------------------------------
class TFIDFEncoder(BaseEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.vectorizer = TfidfVectorizer(
            max_features=cfg.get("max_features", 1000), 
            stop_words=cfg.get("stop_words", 'english'),
            min_df=cfg.get("min_df", 1),
            max_df=cfg.get("max_df", 0.95)
        )
        self._is_fitted = False

    def _fit(self, conversations: pd.Series):
        """
        Fits the vectorizer on the entire corpus of messages.
        Args:
            conversations: A pandas Series containing lists of message dictionaries.
        """
        corpus = []
        # Flatten all messages from all conversations into one list
        for msg_list in conversations:
            for msg in msg_list:
                content = msg.get("content", "")
                if content and isinstance(content, str):
                    corpus.append(content)
        self.vectorizer.fit(corpus)
        self._is_fitted = True

    def _encode(self, conversations: pd.Series) -> list[list[list[float]]]:
        """
        Encodes each conversation into a matrix of embeddings.
        Returns: A list (one per conversation) of lists (one per message) of floats.
        """
        if not self._is_fitted:
            raise ValueError("Encoder must be fit before encoding.")

        results = []
        for msg_list in conversations:
            # Extract text content for this conversation
            texts = [msg.get("content", "") for msg in msg_list]
            
            # Handle empty conversations
            if not texts:
                results.append([])
                continue

            # Transform returns sparse matrix -> dense array -> list
            # Shape: (n_messages, max_features)
            vectors = self.vectorizer.transform(texts).toarray()
            results.append(vectors.tolist())
            
        return results

    def run(self, load_from: str, save_to: str) -> pd.DataFrame:
        df = pd.read_parquet(load_from + "/dataset.parquet")
        self._fit(df["messages"])
        embeddings = self._encode(df["messages"])
        df["embeddings"] = embeddings
        df.to_parquet(save_to + "/embeddings.parquet", index=False, engine="pyarrow")
        return df