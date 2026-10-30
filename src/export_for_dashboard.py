from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE.parent / "data" / "Online Retail.xlsx"
OUT_DIR = BASE / "results"
TOP_N = 5  # recommendations per customer

OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------
print("Loading data...")
df = pd.read_excel(DATA_FILE, engine="openpyxl")

df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]  # cancelled invoices
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
df = df[df["Quantity"] < 50000]  # huge orders that were later cancelled

df["CustomerID"] = df["CustomerID"].astype(int)
df["StockCode"] = df["StockCode"].astype(str)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# ---------------------------------------------------------------
# 2. RFM
# ---------------------------------------------------------------
print("Computing RFM and segments...")
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalAmount", "sum"),
).reset_index()
rfm["Monetary"] = rfm["Monetary"].round(2)

# Temporary rule-based segment. segment_rfm.py replaces this with K-Means segments.
rfm["Segment"] = "Unassigned"

# ---------------------------------------------------------------
# 3. Recommendations (item-based cosine similarity)
# ---------------------------------------------------------------
print("Building recommendations...")
cp = df.groupby(["CustomerID", "StockCode"])["Quantity"].sum().unstack(fill_value=0)
customers = cp.index.to_numpy()
products = cp.columns.to_numpy()

M = cp.to_numpy(dtype=np.float32)
sim = cosine_similarity(M.T).astype(np.float32)

bought = (M > 0).astype(np.float32)
scores = bought @ sim
scores[bought > 0] = -np.inf  # never recommend something already bought

top_idx = np.argsort(-scores, axis=1)[:, :TOP_N]

names = (
    df.drop_duplicates("StockCode")
    .set_index("StockCode")["Description"]
    .astype(str)
    .str.strip()
)

rows = []
for i, cust in enumerate(customers):
    for rank, j in enumerate(top_idx[i], start=1):
        score = scores[i, j]
        if not np.isfinite(score) or score <= 0:
            continue
        code = products[j]
        rows.append({
            "CustomerID": int(cust),
            "StockCode": code,
            "Description": names.get(code, "Unknown Product"),
            "Score": round(float(score), 4),
            "Rank": rank,
        })

rec = pd.DataFrame(rows)

# ---------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------
rfm.to_csv(OUT_DIR / "rfm.csv", index=False)
rec.to_csv(OUT_DIR / "recommendations.csv", index=False)

print(f"Done: {len(rfm)} customers, {len(rec)} recommendations.")
print(f"Files saved in {OUT_DIR}. Now run segment_rfm.py to add K-Means segments.")