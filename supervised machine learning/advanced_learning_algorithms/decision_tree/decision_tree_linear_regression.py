import numpy as np
import matplotlib.pyplot as plt


# Set the seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 100

# Generate random x values (features)
x_train = 2 * np.random.rand(num_samples, 1)

# Generate corresponding y values with a linear relationship (y = 4 + 3x + noise)
true_slope = 3
true_intercept = 4
noise = np.random.randn(num_samples, 1)

y_train = true_intercept + true_slope * x_train + noise
def variance(y):
    if len(y) == 0:
        return 0
    return np.var(y)

def variance_reduction(y_parent, y_left, y_right):
    w_left = len(y_left) / len(y_parent)
    w_right = len(y_right) / len(y_parent)
    return variance(y_parent) - (w_left * variance(y_left) + w_right * variance(y_right))

def nodecal(X_node, y_node):
    best_feature = None
    best_threshold = None
    best_vr = -1   # variance reduction

    n_features = X_node.shape[1]

    for i in range(n_features):
        thresholds = X_node[:, i]

        for threshold in thresholds:
            left_y = []
            right_y = []

            for idx in range(len(X_node)):
                if X_node[idx, i] > threshold:
                    left_y.append(y_node[idx])
                else:
                    right_y.append(y_node[idx])

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            vr = variance_reduction(
                y_node,
                np.array(left_y),
                np.array(right_y)
            )

            if vr > best_vr:
                best_vr = vr
                best_feature = i
                best_threshold = threshold

    return best_feature, best_threshold, best_vr

def build_tree(X, y, depth=0, max_depth=3):
    # stopping conditions
    if len(y) == 0 or depth == max_depth:

        return {
            "type": "leaf",
            "value": np.mean(y)

        }

    feature, threshold, ig = nodecal(X, y)

    if feature is None or ig == 0:
        return {
            "type": "leaf",
            "value": np.mean(y)

        }

    # split data
    left_idx = X[:, feature] > threshold
    right_idx = X[:, feature] <= threshold

    left_subtree = build_tree(
        X[left_idx], y[left_idx], depth + 1, max_depth
    )
    right_subtree = build_tree(
        X[right_idx], y[right_idx], depth + 1, max_depth
    )

    return {
        "type": "node",
        "feature": feature,
        "threshold": threshold,
        "left": left_subtree,
        "right": right_subtree
    }

def predict_one(x, tree):
    if tree["type"] == "leaf":
        return tree["value"]

    if x[tree["feature"]] > tree["threshold"]:
        return predict_one(x, tree["left"])
    else:
        return predict_one(x, tree["right"])
X = x_train
y = y_train.flatten()

tree = build_tree(X, y, max_depth=3)
y_pred = np.array([predict_one(x, tree) for x in X])


y_pred = np.array([predict_one(x, tree) for x in X])
mse = np.mean((y_pred - y) ** 2)
print("MSE:", mse)
# sort x for clean staircase plot
sorted_idx = np.argsort(x_train[:, 0])
x_sorted = x_train[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]


# Plotting the data
plt.figure(figsize=(8, 5))

# scatter plot
plt.scatter(x_train, y_train, alpha=0.6, label="Training data")

# regression tree staircase
plt.step(
    x_sorted.flatten(),
    y_pred_sorted,
    where="post",
    linewidth=3,
    label="Decision Tree Prediction"
)

plt.xlabel("x_train")
plt.ylabel("y_train")
plt.title("Regression Tree Decision Boundary (Staircase)")
plt.legend()
plt.grid(True)
plt.show()
