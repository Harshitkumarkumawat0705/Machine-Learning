import numpy as np
from matplotlib import pyplot as plt


def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5  # 12-15 min is best
    X[:, 0] = X[:, 0] * (285 - 150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        y = -3 / (260 - 175) * t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1
    return (X, Y.reshape(-1, 1))


X, y = load_coffee_data()
print(X.shape, y.shape)
y=y.flatten() 
X1= X[:,0]
X2=X[:,1]
def pro(lables):
    if len(lables) == 0:
        return 0  
    count = 0
    for i in range (len(lables)):
        if(lables[i]==1):
            count +=1
    return count/len(lables)
def weight(L1,L2):
    return len(L1)/(len(L1)+len(L2))
def entropy(p):
    if p == 0 or p == 1:
        return 0
    return (( -p*np.log2(p) ) - ( (1-p)*np.log2(1-p) ) )
def information_gain(W1,W2,h1,h2,H):
    return H-(W1*h1+W2*h2)
def nodecal(X_node, y_node):
    best_feature = None
    best_threshold = None
    best_ig = -1

    P = pro(y_node)
    H = entropy(P)

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

            p_left = pro(left_y)
            p_right = pro(right_y)

            h_left = entropy(p_left)
            h_right = entropy(p_right)

            w_left = len(left_y) / len(y_node)
            w_right = len(right_y) / len(y_node)

            ig = information_gain(w_left, w_right, h_left, h_right, H)

            if ig > best_ig:
                best_ig = ig
                best_feature = i
                best_threshold = threshold

    return best_feature, best_threshold, best_ig
def build_tree(X, y, depth=0, max_depth=3):
    # stopping conditions
    if len(y) == 0 or np.all(y == y[0]) or depth == max_depth:
        return {
            "type": "leaf",
            "value": int(round(pro(y)))
        }

    feature, threshold, ig = nodecal(X, y)

    if feature is None or ig == 0:
        return {
            "type": "leaf",
            "value": int(round(pro(y)))
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


tree = build_tree(X, y, max_depth=3)
print("Tree built successfully")
def predict_one(x, tree):
    if tree["type"] == "leaf":
        return tree["value"]

    if x[tree["feature"]] > tree["threshold"]:
        return predict_one(x, tree["left"])
    else:
        return predict_one(x, tree["right"])


y_pred = np.array([predict_one(x, tree) for x in X])
print("Accuracy:", np.mean(y_pred == y))


# Plot
# plt.figure(figsize=(8, 6))
# plt.scatter(X[y.flatten() == 0][:, 0], X[y.flatten() == 0][:, 1], color="red", label="Class 0", alpha=0.6)
# plt.scatter(X[y.flatten() == 1][:, 0], X[y.flatten() == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
# # Decision boundary
# if tree["type"] == "node":
#     if tree["feature"] == 0:
#         plt.axvline(x=tree["threshold"], linestyle='--', linewidth=2, label='Root Split')
#     else:
#         plt.axhline(y=tree["threshold"], linestyle='--', linewidth=2, label='Root Split')

# plt.title("Dummy Binary Classification Data (for Logistic Regression)")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.legend()
# plt.grid(True)
# plt.show()
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = np.array([predict_one(p, tree) for p in grid])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[y == 0][:,0], X[y == 0][:,1], c="red")
plt.scatter(X[y == 1][:,0], X[y == 1][:,1], c="blue")
plt.show()

