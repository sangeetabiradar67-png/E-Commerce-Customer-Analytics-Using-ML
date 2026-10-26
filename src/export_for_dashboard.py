from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
CANDIDATES = [
    BASE.parent / "data" / "Online Retail.xlsx",   # your layout: repo/data
    BASE / "data" / "Online Retail.xlsx",
    BASE / "Online Retail.xlsx",
]
DATA_FILE = next((p for p in CANDIDATES if p.exists()), CANDIDATES[0])
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)
TOP_PRODUCTS = 1000      # limit the product catalogue used for recommendations
RECS_PER_CUSTOMER = 5
N_CLUSTERS = 4


def load_clean(path):
    df = pd.read_excel(path, engine="openpyxl")
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]   # cancellations
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_rfm(df):
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda d: (snapshot - d.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    return rfm.round({"Monetary": 2})


def add_segments(rfm):
    features = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    scaled = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42).fit_predict(scaled)

    # Name clusters by overall value: low recency + high frequency/spend is best.
    z = pd.DataFrame(scaled, index=rfm.index, columns=["R", "F", "M"])
    z["cluster"] = labels
    value = z.groupby("cluster").apply(lambda g: (-g["R"] + g["F"] + g["M"]).mean())
    names = ["Lost", "At risk", "Loyal customers", "Champions"]
    mapping = {c: names[i] for i, c in enumerate(value.sort_values().index)}
    rfm["Segment"] = pd.Series(labels, index=rfm.index).map(mapping)
    return rfm


def build_recommendations(df):
    top = df["StockCode"].value_counts().head(TOP_PRODUCTS).index
    sub = df[df["StockCode"].isin(top)]

    matrix = (
        sub.assign(bought=1)
        .pivot_table(index="CustomerID", columns="StockCode", values="bought",
                     aggfunc="max", fill_value=0)
    )
    item_sim = cosine_similarity(matrix.T.values)
    scores = matrix.values @ item_sim
    scores[matrix.values == 1] = -1          # never recommend what they already bought

    descriptions = (
        df.groupby("StockCode")["Description"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "")
    )

    rows = []
    stock_codes = matrix.columns.to_numpy()
    for i, customer in enumerate(matrix.index):
        best = np.argsort(scores[i])[::-1][:RECS_PER_CUSTOMER]
        max_score = scores[i][best[0]] or 1
        for j in best:
            if scores[i][j] <= 0:
                continue
            code = stock_codes[j]
            rows.append({
                "CustomerID": customer,
                "StockCode": code,
                "Description": str(descriptions.get(code, "")).strip().replace(",", " "),
                "Score": round(float(scores[i][j] / max_score), 4),
            })
    return pd.DataFrame(rows)


def main():
    if not DATA_FILE.exists():
        raise SystemExit(f"Could not find Online Retail.xlsx. Looked in: {[str(p) for p in CANDIDATES]}")

    print("Loading data...")
    df = load_clean(DATA_FILE)

    print("Computing RFM and segments...")
    rfm = add_segments(build_rfm(df))
    rfm.reset_index().to_csv(OUT_DIR / "rfm.csv", index=False)

    print("Building recommendations...")
    recs = build_recommendations(df)
    recs.to_csv(OUT_DIR / "recommendations.csv", index=False)

    print(f"Done: {len(rfm)} customers, {len(recs)} recommendations.")
    print(f"Files saved in {OUT_DIR}. Open index.html and choose rfm.csv and recommendations.csv.")


if __name__ == "__main__":
    main()