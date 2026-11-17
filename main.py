import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Online Retail dataset
df = pd.read_excel("../data/Online Retail.xlsx", engine="openpyxl")

print("First 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# ---------------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------------
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
df = df[df["Quantity"] < 50000]
df["CustomerID"] = df["CustomerID"].astype(int)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("\nCleaned dataset shape:")
print(df.shape)

# ---------------------------------------------------------------
# BASIC STATISTICS
# ---------------------------------------------------------------
# One revenue column used everywhere (was duplicated as TotalPrice and TotalAmount)
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print("\nTotal Revenue:", df["TotalPrice"].sum())
print("Total Customers:", df["CustomerID"].nunique())
print("Total Products:", df["StockCode"].nunique())
print("Total Transactions:", df["InvoiceNo"].nunique())

top_customers = (
    df.groupby("CustomerID")["TotalPrice"].sum().sort_values(ascending=False).head(10)
)
print("\nTop 10 Customers by Spending:")
print(top_customers)

top_products = (
    df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
)
print("\nTop 10 Most Purchased Products:")
print(top_products)

# ---------------------------------------------------------------
# POPULAR PRODUCTS AND TOP CUSTOMERS
# ---------------------------------------------------------------
customer_product = df.pivot_table(
    index="CustomerID",
    columns="StockCode",
    values="Quantity",
    aggfunc="sum",
    fill_value=0,
)

popular_products = customer_product.sum(axis=0).sort_values(ascending=False).head(10)
print("\nTop 10 Most Popular Products:")
for product, quantity in popular_products.items():
    print(product, "-", quantity)

plt.figure(figsize=(12, 6))
popular_products.sort_values().plot(kind="barh")
plt.title("Top 10 Most Popular Products")
plt.xlabel("Quantity Sold")
plt.ylabel("Product ID")
plt.tight_layout()
plt.savefig("../results/top_products.png")
plt.close()

top_customers_qty = customer_product.sum(axis=1).sort_values(ascending=False).head(10)
print("\nTop 10 Customers by Purchase Quantity:")
print(top_customers_qty)

plt.figure(figsize=(10, 6))
top_customers_qty.sort_values().plot(kind="barh")
plt.title("Top 10 Customers by Purchase Quantity")
plt.xlabel("Total Purchase Quantity")
plt.ylabel("Customer ID")
plt.tight_layout()
plt.savefig("../results/top_customers.png")
plt.close()

customer_product_count = (customer_product > 0).sum(axis=1)
print("\nCustomer Purchase Pattern:")
print("Average different products purchased:", round(customer_product_count.mean(), 2))
print("Maximum different products purchased:", customer_product_count.max())
print("Minimum different products purchased:", customer_product_count.min())

# ---------------------------------------------------------------
# RFM ANALYSIS (moved before segmentation, segmentation needs it)
# ---------------------------------------------------------------
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
    TotalQuantity=("Quantity", "sum"),
).reset_index()

print("\nRFM Analysis Results:")
print(rfm.head(10))

# ---------------------------------------------------------------
# CUSTOMER SEGMENTATION (FIXED): K-Means on RFM instead of qcut
# ---------------------------------------------------------------
# FIX: pd.qcut always makes 3 equal-sized groups, so every bar was ~1,440.
# K-Means groups customers by real behaviour, so sizes can differ.
features = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])  # log reduces skew
X = StandardScaler().fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X)

# Name clusters by average spending: lowest -> Low, highest -> High
rank = rfm.groupby("Cluster")["Monetary"].mean().sort_values().index.tolist()
names = {rank[0]: "Low Purchase", rank[1]: "Medium Purchase", rank[2]: "High Purchase"}
rfm["Segment"] = rfm["Cluster"].map(names)

order = ["Low Purchase", "Medium Purchase", "High Purchase"]

print("\nSegment summary:")
summary = rfm.groupby("Segment").agg(
    Customers=("Segment", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean"),
).reindex(order).round(2)
print(summary)

# FIX: reindex(order) so bars always go Low, Medium, High
# (value_counts() sorts by size, which gave Low, High, Medium)
segment_counts = rfm["Segment"].value_counts().reindex(order)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(order, segment_counts.values, color=["#C62828", "#F9A825", "#2E7D32"], width=0.5)
ax.bar_label(ax.containers[0], padding=3)  # FIX: counts on top of bars
ax.set_title("Customer Segmentation Based on RFM (K-Means)")
ax.set_xlabel("Customer Segment")
ax.set_ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("../results/customer_segmentation.png", dpi=150)
plt.close()

# Save results
rfm.to_csv("../results/customer_rfm.csv", index=False)
print("\nRFM analysis and segments saved to results/customer_rfm.csv")