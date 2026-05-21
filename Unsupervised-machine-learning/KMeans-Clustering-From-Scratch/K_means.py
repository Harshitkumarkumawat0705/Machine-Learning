import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Set seed
np.random.seed(42)

# Generate a classification dataset
x, y = make_classification(
    n_samples=200,
    n_features=2,          # Only 2 useful features for visualization
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,         # Low separation makes it harder
    flip_y=0.1,            # Add noise (10% label flipping)
    random_state=42
)
k = 2
index = np.random.choice(x.shape[0], size=k, replace=False)
centroids=x[index]
costs = []
for _ in range (20):
    dist = np.sqrt(((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
    labels = np.argmin(dist, axis =1)
    new_centroids = []
    for i in range(k):
        cluster_mean = x[labels == i].mean(axis=0)
        new_centroids.append(cluster_mean)
    centroids = np.array(new_centroids)
    assigned_distances = dist[np.arange(len(x)), labels]
    cost = np.sum(assigned_distances**2)
    costs.append(cost)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.title("Dummy Binary Classification Data (for Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.title("Dummy Binary Classification Data (for Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="X", s=200, label="Centroids")
plt.show()
plt.figure(figsize=(8, 4))
plt.plot(range(len(costs)), costs, marker='o', color='green', linestyle='--')
plt.title("K-Means Cost (Inertia) Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Sum of Squared Distances")
plt.grid(True)
plt.show()

print("Features shape:", x.shape)
print("Target shape:", y.shape)

np.random.seed(None)