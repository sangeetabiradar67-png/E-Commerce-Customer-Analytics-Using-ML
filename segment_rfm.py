import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RFM_FILE = "results/rfm.csv"
SUMMARY_FILE = "results/segment_summary.csv"
CHART_FILE = "results/segment_chart.png"
K = 4  # chosen for 4 easy-to-explain business segments (see silhouette scores below)

rfm = pd.read_csv(RFM_FILE)

# Log-transform to tame skew (a few huge spenders), then scale.
# clip(lower=0) stops negative Monetary (returns) from breaking log1p.
X = np.log1p(rfm[["Recency", "Frequency", "Monetary"]].clip(lower=0))
scaled = StandardScaler().fit_transform(X)

# Compare k values (higher silhouette = better separated clusters)
print("k  silhouette")
for k in range(2, 8):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(scaled)
    print(f"{k}  {silhouette_score(scaled, labels):.3f}")

km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(scaled)
rfm["Cluster"] = km.labels_

# Rank clusters: low recency is good; high frequency and monetary are good
centers = pd.DataFrame(km.cluster_centers_, columns=["R", "F", "M"])
centers["score"] = -centers["R"] + centers["F"] + centers["M"]
ranked = centers["score"].sort_values(ascending=False).index.tolist()

# Names go from best to worst cluster
names = ["Champions", "Loyal", "New / Promising", "Lost"] if K == 4 else [f"Segment {i+1}" for i in range(K)]
rfm["Segment"] = rfm["Cluster"].map({c: names[i] for i, c in enumerate(ranked)})

summary = (
    rfm.groupby("Segment")
    .agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        TotalRevenue=("Monetary", "sum"),
    )
    .reindex(names)  # keep best -> worst order in the table too
    .round(1)
)
summary["RevenueShare%"] = (100 * summary["TotalRevenue"] / summary["TotalRevenue"].sum()).round(1)
print(summary)

rfm.drop(columns="Cluster").to_csv(RFM_FILE, index=False)
summary.to_csv(SUMMARY_FILE)
print("Saved Segment column to", RFM_FILE)

# ---------------- Chart ----------------
SEGMENT_COLORS = {
    "Champions": "#2E7D32",
    "Loyal": "#1565C0",
    "New / Promising": "#F9A825",
    "Lost": "#C62828",
}

# FIX: keep best -> worst order instead of sorting by size
counts = summary["Customers"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    counts.index,
    counts.values,
    color=[SEGMENT_COLORS.get(s, "#888888") for s in counts.index],
    width=0.5,
)
ax.bar_label(ax.containers[0], padding=3)  # FIX: show counts on bars
ax.set_title("Customers per Segment (RFM + K-Means)")
ax.set_xlabel("Segment")
ax.set_ylabel("Number of customers")
plt.tight_layout()
plt.savefig(CHART_FILE, dpi=150)
plt.close()
print("Saved", CHART_FILE)