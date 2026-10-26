import matplotlib
import os
os.makedirs("../results", exist_ok=True)
matplotlib.use("Agg")  # save charts to files instead of opening windows (so the script doesn't pause)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load Online Retail dataset
df = pd.read_excel("../data/Online Retail.xlsx", engine="openpyxl")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset shape
print("\nDataset shape:")
print(df.shape)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# ---------------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------------

# Remove rows where CustomerID is missing
df = df.dropna(subset=["CustomerID"])

# Remove cancelled transactions
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Remove invalid quantities
df = df[df["Quantity"] > 0]

# Remove invalid prices
df = df[df["UnitPrice"] > 0]

# Remove huge orders that were later cancelled (e.g. customer 12346)
df = df[df["Quantity"] < 50000]

# Convert CustomerID to integer
df["CustomerID"] = df["CustomerID"].astype(int)

# Check cleaned dataset
print("\nCleaned dataset shape:")
print(df.shape)

print("\nFirst 5 rows after cleaning:")
print(df.head())

# ---------------------------------------------------------------
# BASIC STATISTICS
# ---------------------------------------------------------------

# Calculate total revenue
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print("\nTotal Revenue:")
print(df["TotalPrice"].sum())

# Total number of customers
print("\nTotal Customers:")
print(df["CustomerID"].nunique())

# Total number of products
print("\nTotal Products:")
print(df["StockCode"].nunique())

# Total number of transactions
print("\nTotal Transactions:")
print(df["InvoiceNo"].nunique())

# Top 10 customers by total spending
top_customers = (
    df.groupby("CustomerID")["TotalPrice"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Customers by Spending:")
print(top_customers)

# Top 10 most purchased products
top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Most Purchased Products:")
print(top_products)

# ---------------------------------------------------------------
# RECOMMENDATION SYSTEM (item-based, cosine similarity)
# ---------------------------------------------------------------

# Create Customer-Product Matrix
customer_product = df.pivot_table(
    index="CustomerID",
    columns="StockCode",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

print("\nCustomer-Product Matrix Shape:")
print(customer_product.shape)

print("\nCustomer-Product Matrix:")
print(customer_product.head())

# Calculate product similarity
product_similarity = cosine_similarity(customer_product.T)

print("\nProduct Similarity Matrix Shape:")
print(product_similarity.shape)

# Create product index mapping
product_indices = pd.Series(
    range(len(customer_product.columns)),
    index=customer_product.columns
)


# Recommendation function
def recommend_for_customer(customer_id, num_recommendations=5):

    # Check whether customer exists
    if customer_id not in customer_product.index:
        return "Customer not found"

    # Get products purchased by the customer
    purchased_products = customer_product.loc[customer_id]

    purchased_products = purchased_products[
        purchased_products > 0
    ].index

    # Store recommendation scores
    recommendation_scores = {}

    # Compare each purchased product with other products
    for product in purchased_products:

        product_index = product_indices[product]

        similarity_scores = product_similarity[product_index]

        for i, score in enumerate(similarity_scores):

            recommended_product = customer_product.columns[i]

            # Don't recommend products already purchased
            if recommended_product not in purchased_products:

                if recommended_product not in recommendation_scores:
                    recommendation_scores[recommended_product] = 0

                recommendation_scores[recommended_product] += score

    # Sort recommendations
    recommendations = sorted(
        recommendation_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return top recommendations
    return recommendations[:num_recommendations]


# Create StockCode to Product Name mapping
product_names = (
    df[["StockCode", "Description"]]
    .drop_duplicates("StockCode")
    .set_index("StockCode")["Description"]
)

# Test recommendation system
customer_id = 17850

print("\nRecommended Products for Customer", customer_id)

recommendations = recommend_for_customer(customer_id, 5)

for product_code, score in recommendations:
    product_name = product_names.get(product_code, "Unknown Product")
    print(f"{product_code} - {product_name} (Score: {score:.2f})")

# NOTE: Hit Rate evaluation now lives in evaluate_recommender.py

# ---------------------------------------------------------------
# POPULAR PRODUCTS AND TOP CUSTOMERS
# ---------------------------------------------------------------

# Top 10 most popular products
popular_products = customer_product.sum(axis=0).sort_values(
    ascending=False
).head(10)

print("\nTop 10 Most Popular Products:")

for product, quantity in popular_products.items():
    print(product, "-", quantity)

# Visualize the Top 10 Most Popular Products
plt.figure(figsize=(12, 6))
popular_products.sort_values().plot(kind="barh")
plt.title("Top 10 Most Popular Products")
plt.xlabel("Quantity Sold")
plt.ylabel("Product ID")
plt.tight_layout()
plt.savefig("../results/top_products.png")
plt.close()

# Top 10 customers by total purchase quantity
top_customers_qty = customer_product.sum(axis=1).sort_values(
    ascending=False
).head(10)

print("\nTop 10 Customers by Purchase Quantity:")
print(top_customers_qty)

# Visualize top 10 customers
plt.figure(figsize=(10, 6))
top_customers_qty.sort_values().plot(kind="barh")
plt.title("Top 10 Customers by Purchase Quantity")
plt.xlabel("Total Purchase Quantity")
plt.ylabel("Customer ID")
plt.tight_layout()
plt.savefig("../results/top_customers.png")
plt.close()

# Analyze how many different products each customer purchased
customer_product_count = (customer_product > 0).sum(axis=1)

print("\nCustomer Purchase Pattern:")
print("Average different products purchased:",
      round(customer_product_count.mean(), 2))

print("Maximum different products purchased:",
      customer_product_count.max())

print("Minimum different products purchased:",
      customer_product_count.min())

# ---------------------------------------------------------------
# SEGMENTATION BY PURCHASE QUANTITY
# ---------------------------------------------------------------

# Calculate total quantity purchased by each customer
customer_quantity = df.groupby("CustomerID")["Quantity"].sum().reset_index()

# Rename columns
customer_quantity.columns = ["CustomerID", "TotalQuantity"]

# Divide customers into three segments
customer_quantity["Segment"] = pd.qcut(
    customer_quantity["TotalQuantity"],
    q=3,
    labels=["Low Purchase", "Medium Purchase", "High Purchase"],
    duplicates="drop"
)

# Display customer segments
print("\nCustomer Segmentation Results:")
print(customer_quantity.head(10))

# Count customers in each segment
print("\nNumber of customers in each segment:")
print(customer_quantity["Segment"].value_counts())

# Visualize customer segments
segment_counts = customer_quantity["Segment"].value_counts()

plt.figure(figsize=(8, 5))
segment_counts.plot(kind="bar")
plt.title("Customer Segmentation Based on Purchase Quantity")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()

# Save the chart
plt.savefig("../results/customer_segmentation.png")
plt.close()

# ---------------------------------------------------------------
# RFM ANALYSIS
# ---------------------------------------------------------------

# Calculate total amount for each transaction
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# Set the reference date to one day after the latest transaction
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Calculate RFM metrics for each customer
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalAmount", "sum")
).reset_index()

# Display RFM results
print("\nRFM Analysis Results:")
print(rfm.head(10))

# Save the RFM results
rfm.to_csv("../results/customer_rfm.csv", index=False)

print("\nRFM analysis saved to results/customer_rfm.csv")