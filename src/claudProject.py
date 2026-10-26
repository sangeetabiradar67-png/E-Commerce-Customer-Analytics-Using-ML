"""
Item-based collaborative filtering recommender with RFM-segment fallback.

Run rfm_segmentation.py first so results/rfm_segments.csv exists
(the script still works without it, using overall popularity as fallback).

Usage (PowerShell, from the project root):
    python src\\customer_analytics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from rfm_segmentation import DEFAULT_FILE, RESULTS_DIR, clean_data, load_data

SEGMENTS_FILE = RESULTS_DIR / "rfm_segments.csv"
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Data prep
# --------------------------------------------------------------------------
def keep_products_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-product stock codes (POST, D, M, BANK CHARGES, ...)."""
    df = df.copy()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    return df[df["StockCode"].str.match(r"^\d{5}")]


def product_names(df: pd.DataFrame) -> pd.Series:
    return (df.dropna(subset=["Description"])
              .assign(Description=lambda d: d["Description"].astype(str).str.strip())
              .groupby("StockCode")["Description"].first())


# --------------------------------------------------------------------------
# Recommender
# --------------------------------------------------------------------------
class Recommender:
    def __init__(self, df: pd.DataFrame, cust_idx=None, item_idx=None,
                 names: pd.Series = None, segments: pd.Series = None):
        self.cust_idx = cust_idx or {c: i for i, c in enumerate(sorted(df["CustomerID"].unique()))}
        self.item_idx = item_idx or {s: i for i, s in enumerate(sorted(df["StockCode"].unique()))}
        self.items = list(self.item_idx.keys())
        self.names = names if names is not None else product_names(df)

        pairs = df[["CustomerID", "StockCode"]].drop_duplicates()
        rows = pairs["CustomerID"].map(self.cust_idx).to_numpy()
        cols = pairs["StockCode"].map(self.item_idx).to_numpy()
        self.M = csr_matrix((np.ones(len(pairs)), (rows, cols)),
                            shape=(len(self.cust_idx), len(self.item_idx)))

        self.sim = cosine_similarity(self.M.T.tocsr(), dense_output=True)
        np.fill_diagonal(self.sim, 0.0)
        self.popularity = np.asarray(self.M.sum(axis=0)).ravel()

        # Popular items per RFM segment (optional)
        self.segments = segments
        self.segment_pop = {}
        if segments is not None:
            seg_of_row = pd.Series(segments).reindex(list(self.cust_idx.keys()))
            for seg in seg_of_row.dropna().unique():
                mask = (seg_of_row == seg).to_numpy()
                self.segment_pop[seg] = np.asarray(self.M[mask].sum(axis=0)).ravel()

    # -- scoring ----------------------------------------------------------
    def _scores(self, customer_id, n: int) -> np.ndarray:
        """Return item scores for a customer, with already-bought items masked out."""
        if customer_id not in self.cust_idx:
            return self.popularity.astype(float).copy()

        u = self.cust_idx[customer_id]
        bought = self.M[u].indices
        scores = np.asarray(self.M[u].dot(self.sim)).ravel()

        # Fill with segment-popular (else overall popular) items if signal is thin
        if (scores > 0).sum() < n:
            seg = self.segments.get(customer_id) if self.segments is not None else None
            fallback = self.segment_pop.get(seg, self.popularity).astype(float)
            fallback = fallback / (fallback.max() or 1.0) * 1e-3   # always ranks below real scores
            scores = scores + fallback

        scores[bought] = -np.inf
        return scores

    def _top_n(self, scores: np.ndarray, n: int) -> np.ndarray:
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        return top[np.argsort(-scores[top])]

    def recommend(self, customer_id, n: int = 5) -> pd.DataFrame:
        scores = self._scores(customer_id, n)
        top = self._top_n(scores, n)
        codes = [self.items[i] for i in top]
        return pd.DataFrame({
            "StockCode": codes,
            "Description": [self.names.get(c, "") for c in codes],
            "Score": np.round(scores[top], 4),
        })

    def similar_items(self, stock_code: str, n: int = 5) -> pd.DataFrame:
        i = self.item_idx[str(stock_code).upper()]
        top = self._top_n(self.sim[i], n)
        codes = [self.items[j] for j in top]
        return pd.DataFrame({
            "StockCode": codes,
            "Description": [self.names.get(c, "") for c in codes],
            "Similarity": np.round(self.sim[i][top], 4),
        })


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_recommendation_system(df: pd.DataFrame, segments=None, k: int = 10,
                                   test_frac: float = 0.2, min_items: int = 5,
                                   max_customers: int = 1000) -> pd.DataFrame:
    """Hold out part of each customer's products, then check if we recommend them."""
    rng = np.random.default_rng(RANDOM_STATE)
    pairs = df[["CustomerID", "StockCode"]].drop_duplicates()
    counts = pairs.groupby("CustomerID")["StockCode"].transform("size")
    pairs = pairs[counts >= min_items]

    is_test = rng.random(len(pairs)) < test_frac
    train, test = pairs[~is_test], pairs[is_test]

    # Shared indices so train and full matrices line up
    cust_idx = {c: i for i, c in enumerate(sorted(pairs["CustomerID"].unique()))}
    item_idx = {s: i for i, s in enumerate(sorted(df["StockCode"].unique()))}
    model = Recommender(train, cust_idx, item_idx, names=product_names(df), segments=segments)

    test_items = test.groupby("CustomerID")["StockCode"].apply(set)
    sample = test_items.sample(min(max_customers, len(test_items)), random_state=RANDOM_STATE)

    results = {"Item-based CF": [], "Popularity baseline": []}
    for cust, held_out in sample.items():
        # Model
        scores = model._scores(cust, k)
        rec = {model.items[i] for i in model._top_n(scores, k)}
        results["Item-based CF"].append(_metrics(rec, held_out, k))

        # Baseline: most popular items the customer hasn't bought
        pop = model.popularity.astype(float).copy()
        pop[model.M[cust_idx[cust]].indices] = -np.inf
        base = {model.items[i] for i in model._top_n(pop, k)}
        results["Popularity baseline"].append(_metrics(base, held_out, k))

    rows = {name: pd.DataFrame(vals).mean() for name, vals in results.items()}
    out = pd.DataFrame(rows).T.round(4)
    out.columns = [f"HitRate@{k}", f"Precision@{k}", f"Recall@{k}"]
    print(f"Evaluated on {len(sample)} customers (min {min_items} products each)")
    return out


def _metrics(recommended: set, held_out: set, k: int) -> dict:
    hits = len(recommended & held_out)
    return {"hit": float(hits > 0), "precision": hits / k, "recall": hits / len(held_out)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    df = keep_products_only(clean_data(load_data(path)))

    segments = None
    if SEGMENTS_FILE.exists():
        segments = pd.read_csv(SEGMENTS_FILE, index_col="CustomerID")["Segment"]
        print(f"Loaded segments for {len(segments):,} customers")
    else:
        print("rfm_segments.csv not found - using overall popularity as fallback")

    print("\nBuilding recommender ...")
    model = Recommender(df, segments=segments)

    # Sample outputs
    sample_customer = sorted(model.cust_idx)[0]
    print(f"\nTop 5 recommendations for customer {sample_customer}:")
    print(model.recommend(sample_customer, n=5).to_string(index=False))

    sample_item = model.items[0]
    print(f"\nProducts similar to {sample_item} ({model.names.get(sample_item, '')}):")
    print(model.similar_items(sample_item, n=5).to_string(index=False))

    print("\nOffline evaluation:")
    print(evaluate_recommendation_system(df, segments=segments, k=10).to_string())


if __name__ == "__main__":
    main()