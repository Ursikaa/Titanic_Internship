import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load cleaned Titanic dataset
df = pd.read_csv("cleaned_titanic.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# 2. Select numerical features for clustering
features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]

X = df[features].copy()

# 3. Handle any missing values
X = X.fillna(X.median())

print("\nMissing values after cleaning:")
print(X.isnull().sum())

# 4. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Elbow Method
inertia = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.show()
# 6. Apply K-Means with K = 4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 7. Display cluster sizes
print("\nCluster Sizes:")
print(df["Cluster"].value_counts().sort_index())

# 8. Calculate cluster-wise averages
cluster_summary = df.groupby("Cluster")[features].mean()

print("\nCluster-wise Average Values:")
print(cluster_summary)
# 6. Apply K-Means with K = 4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 7. Display cluster sizes
print("\nCluster Sizes:")
print(df["Cluster"].value_counts().sort_index())

# 8. Calculate cluster-wise averages
cluster_summary = df.groupby("Cluster")[features].mean()

print("\nCluster-wise Average Values:")
print(cluster_summary)

# 9. Save clustered dataset
df.to_csv("titanic_clustered.csv", index=False)

print("\nClustered dataset saved as titanic_clustered.csv")

# 9. Save clustered dataset
df.to_csv("titanic_clustered.csv", index=False)

print("\nClustered dataset saved as titanic_clustered.csv")
print("\n===== EASY TO READ RESULTS =====")

print("\nCluster Sizes:")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster Means:")
print(cluster_summary.round(2).T)
print("\n===== CLUSTER DETAILS =====")

for cluster in sorted(df["Cluster"].unique()):
    print(f"\nCluster {cluster}:")
    for feature in features:
        value = df[df["Cluster"] == cluster][feature].mean()
        print(f"{feature}: {value:.2f}")
        # 10. Cluster Visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    df["Age"],
    df["Fare"],
    c=df["Cluster"],
    cmap="viridis",
    alpha=0.7
)

plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Titanic Passenger Clusters: Age vs Fare")
plt.colorbar(label="Cluster")
plt.grid(True)

plt.savefig("cluster_age_fare.png", dpi=300, bbox_inches="tight")
plt.show()
# 11. Cluster Size Visualization

cluster_counts = df["Cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.xlabel("Cluster")
plt.ylabel("Number of Passengers")
plt.title("Number of Passengers in Each Cluster")

# Add values on top of bars
for i, value in enumerate(cluster_counts.values):
    plt.text(i, value + 10, str(value), ha="center")

plt.grid(axis="y")
plt.tight_layout()

plt.savefig("cluster_size.png", dpi=300, bbox_inches="tight")
plt.show()
# 12. Cluster-wise Average Fare

avg_fare = df.groupby("Cluster")["Fare"].mean()

plt.figure(figsize=(8, 5))

plt.bar(
    avg_fare.index.astype(str),
    avg_fare.values
)

plt.xlabel("Cluster")
plt.ylabel("Average Fare")
plt.title("Average Fare by Cluster")

for i, value in enumerate(avg_fare.values):
    plt.text(i, value + 5, f"{value:.2f}", ha="center")

plt.grid(axis="y")
plt.tight_layout()

plt.savefig("average_fare_by_cluster.png", dpi=300, bbox_inches="tight")
plt.show()
# 13. Cluster-wise Average Age

avg_age = df.groupby("Cluster")["Age"].mean()

plt.figure(figsize=(8, 5))

plt.bar(
    avg_age.index.astype(str),
    avg_age.values
)

plt.xlabel("Cluster")
plt.ylabel("Average Age")
plt.title("Average Age by Cluster")

for i, value in enumerate(avg_age.values):
    plt.text(i, value + 1, f"{value:.2f}", ha="center")

plt.grid(axis="y")
plt.tight_layout()

plt.savefig("average_age_by_cluster.png", dpi=300, bbox_inches="tight")
plt.show()
# 13. Cluster-wise Average Age

avg_age = df.groupby("Cluster")["Age"].mean()

plt.figure(figsize=(8, 5))

plt.bar(
    avg_age.index.astype(str),
    avg_age.values
)

plt.xlabel("Cluster")
plt.ylabel("Average Age")
plt.title("Average Age by Cluster")

for i, value in enumerate(avg_age.values):
    plt.text(i, value + 1, f"{value:.2f}", ha="center")

plt.grid(axis="y")
plt.tight_layout()

plt.savefig("average_age_by_cluster.png", dpi=300, bbox_inches="tight")
plt.show()
# 10. Final Cluster Analysis

print("\n===== FINAL CLUSTER ANALYSIS =====")

for cluster in sorted(df["Cluster"].unique()):
    cluster_data = df[df["Cluster"] == cluster]

    print(f"\nCluster {cluster}:")
    print(f"Passengers: {len(cluster_data)}")
    print(f"Average Age: {cluster_data['Age'].mean():.2f}")
    print(f"Average Pclass: {cluster_data['Pclass'].mean():.2f}")
    print(f"Average SibSp: {cluster_data['SibSp'].mean():.2f}")
    print(f"Average Parch: {cluster_data['Parch'].mean():.2f}")
    print(f"Average Fare: {cluster_data['Fare'].mean():.2f}")
    