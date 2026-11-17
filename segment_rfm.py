import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RFM_FILE = "results/rfm.csv"
K = 4  # change after looking at the silhouette scores below

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

names = ["Champions", "Loyal", "New / Promising", "Lost"] if K == 4 else [f"Segment {i+1}" for i in range(K)]
rfm["Segment"] = rfm["Cluster"].map({c: names[i] for i, c in enumerate(ranked)})

summary = rfm.groupby("Segment").agg(
    Customers=("CustomerID", "count"),
    AvgRecency=("Recency", "mean"),
    AvgFrequency=("Frequency", "mean"),
    AvgMonetary=("Monetary", "mean"),
    TotalRevenue=("Monetary", "sum"),
).round(1)
print(summary)

rfm.drop(columns="Cluster").to_csv(RFM_FILE, index=False)
summary.to_csv("results/segment_summary.csv")
print("Saved Segment column to", RFM_FILE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEGMENT_COLORS = {
    "Champions": "#2E7D32",
    "Loyal": "#1565C0",
    "New / Promising": "#F9A825",
    "Lost": "#C62828",
}

counts = summary["Customers"].sort_values(ascending=False)
plt.figure(figsize=(8, 5))
counts.plot(kind="bar", color=[SEGMENT_COLORS.get(s, "#888888") for s in counts.index])
plt.title("Customers per Segment")
plt.xlabel("Segment")
plt.ylabel("Number of customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/segment_chart.png")
plt.close()
print("Saved results/segment_chart.png")