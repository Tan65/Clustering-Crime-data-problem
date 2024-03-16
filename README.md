# Clustering-Crime-data-problem
Perform Clustering(Hierarchical, Kmeans &amp; DBSCAN) for the crime data and identify the number of clusters formed and draw inferences.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Load the crime data from CSV file
data = pd.read_csv("crime_data.csv")

# Select the numerical features
features = ['Murder', 'Assault', 'UrbanPop', 'Rape']
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
hc_clusters = hc.fit_predict(X_scaled)

# Perform K-means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)

# Perform DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=2)
dbscan_clusters = dbscan.fit_predict(X_scaled)

# Add clustering labels to the DataFrame
data['HC_Cluster'] = hc_clusters
data['KMeans_Cluster'] = kmeans_clusters

# Visualize the clusters using scatter plots
plt.figure(figsize=(18, 6))

# Univariate Analysis - Histograms
for i, feature in enumerate(features, start=1):
    plt.subplot(2, 4, i)
    sns.histplot(data[feature], bins=10, kde=True, color='skyblue')
    plt.title(f"Histogram of {feature}")

# Bivariate Analysis - Scatter Plots with Clusters
plt.subplot(2, 4, 5)
sns.scatterplot(x='Murder', y='Assault', data=data, hue='HC_Cluster', palette='viridis')
plt.title("Hierarchical Clustering")
plt.xlabel("Murder")
plt.ylabel("Assault")

plt.subplot(2, 4, 6)
sns.scatterplot(x='Murder', y='Assault', data=data, hue='KMeans_Cluster', palette='viridis')
plt.title("K-means Clustering")
plt.xlabel("Murder")
plt.ylabel("Assault")

plt.subplot(2, 4, 7)
sns.scatterplot(x='Murder', y='UrbanPop', data=data, hue='HC_Cluster', palette='viridis')
plt.title("Hierarchical Clustering")
plt.xlabel("Murder")
plt.ylabel("UrbanPop")

plt.subplot(2, 4, 8)
sns.scatterplot(x='Murder', y='UrbanPop', data=data, hue='KMeans_Cluster', palette='viridis')
plt.title("K-means Clustering")
plt.xlabel("Murder")
plt.ylabel("UrbanPop")

plt.tight_layout()
plt.show()

# Multivariate Analysis - Pairplot
sns.pairplot(data, vars=features, hue='HC_Cluster', palette='viridis')
plt.suptitle("Pairplot of Crime Data with Hierarchical Clusters", y=1.02)
plt.show()

sns.pairplot(data, vars=features, hue='KMeans_Cluster', palette='viridis')
plt.suptitle("Pairplot of Crime Data with K-means Clusters", y=1.02)
plt.show()
