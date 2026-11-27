"""Turn cleaned free-text answers into TF-IDF features. Uses only numpy/pandas."""
from collections import Counter

import numpy as np
import pandas as pd


def tokenize(text):
    """Split one cleaned text answer into words, e.g. "hot dogs beer" -> ["hot", "dogs", "beer"]."""
    if pd.isna(text):
        return []
    return str(text).split()


class TextVectorizer:
    """
    TF-IDF for one text column. fit() learns the vocabulary and IDF from the
    training split only; transform() counts words (ignoring unseen ones) and
    multiplies by IDF.
    """

    def __init__(self, min_count=3):
        self.min_count = min_count  # keep words that appear in at least this many answers

    def fit(self, texts):
        # Document frequency: number of answers that contain each word
        doc_freq = Counter(w for t in texts for w in set(tokenize(t)))
        self.vocab = sorted(w for w, n in doc_freq.items() if n >= self.min_count)
        self.index = {w: i for i, w in enumerate(self.vocab)}

        # Smoothed IDF (same formula as sklearn): rarer words get larger weights
        n = len(texts)
        df = np.array([doc_freq[w] for w in self.vocab], dtype=float)
        self.idf = np.log((1 + n) / (1 + df)) + 1
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), len(self.vocab)))
        for i, t in enumerate(texts):
            for w in tokenize(t):
                j = self.index.get(w)
                if j is not None:  # unseen words are ignored
                    X[i, j] += 1
        return X * self.idf
