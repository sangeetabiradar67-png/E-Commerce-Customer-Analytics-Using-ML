from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path

SEED = 42
KS = [5, 10]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "online retail.xlsx"
OUT_FILE = PROJECT_ROOT / "src" / "results" / "recommendation_evaluation.json"

# 1. Load dataset
df = pd.read_excel(DATA_FILE, engine="openpyxl")
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df = df[df["Quantity"] < 50000]  # drop the huge cancelled orders
df["CustomerID"] = df["CustomerID"].astype(int)
df["StockCode"] = df["StockCode"].astype(str)

# 2. Customer x product quantity matrix
cp = df.groupby(["CustomerID", "StockCode"])["Quantity"].sum().unstack(fill_value=0)
M = cp.to_numpy(dtype=np.float32)

# 3. Hide one random purchased product for every customer with 2+ products
rng = np.random.default_rng(SEED)
bought = M > 0
eligible = np.where(bought.sum(axis=1) >= 2)[0]
hidden = np.array([rng.choice(np.flatnonzero(bought[i])) for i in eligible])

train = M.copy()
train[eligible, hidden] = 0  # hidden purchases are removed from ALL training data

# 4. Similarity from training data only (no leakage)
sim = cosine_similarity(train.T).astype(np.float32)

# 5. Score every product for each test customer
B = (train > 0).astype(np.float32)
Be = B[eligible]
already = Be > 0

model_scores = Be @ sim          # sum of similarities to purchased products
model_scores[already] = -np.inf

popularity = B.sum(axis=0)       # number of buyers per product
pop_scores = np.tile(popularity, (len(eligible), 1))
pop_scores[already] = -np.inf


def hit_rates(scores):
    order = np.argsort(-scores, axis=1)[:, : max(KS)]
    return {
        f"hit_rate@{k}": round(float((order[:, :k] == hidden[:, None]).any(axis=1).mean()), 4)
        for k in KS
    }


results = {
    "customers_evaluated": int(len(eligible)),
    "products": int(M.shape[1]),
    "seed": SEED,
    "item_based_cosine": hit_rates(model_scores),
    "popularity_baseline": hit_rates(pop_scores),
}

print(json.dumps(results, indent=2))
OUT_FILE.parent.mkdir(exist_ok=True)
OUT_FILE.write_text(json.dumps(results, indent=2))
print("Saved", OUT_FILE)