import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("animal_features.csv")

X_df = pd.get_dummies(data[["EarShape", "FaceShape", "Whiskers"]])
y = data["Animal"].values

X = X_df.values
feature_names = list(X_df.columns)


def pro1(labels):
    return sum(labels == "Cat") / len(labels)


def weight(l1, l2):
    return len(l1) / (len(l1) + len(l2))


def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p*np.log2(p) - (1-p)*np.log2(1-p)


def info_gain(H, H1, H2, W1, W2):
    return H - (W1 * H1 + W2 * H2)


def nodecal(X, y):

    MAX = -1
    best_feature = None
    best_value = None

    best_LX, best_RX = None, None
    best_Ly, best_Ry = None, None

    p = pro1(y)
    H = entropy(p)

    print("\nCURRENT NODE:")
    print("Labels:", y)

    for f in range(X.shape[1]):

        values = np.unique(X[:, f])

        for val in values:

            left_mask = X[:, f] == val
            right_mask = X[:, f] != val

            c1 = y[left_mask]
            c2 = y[right_mask]

            if len(c1) == 0 or len(c2) == 0:
                continue

            p1 = pro1(c1)
            p2 = pro1(c2)

            h1 = entropy(p1)
            h2 = entropy(p2)

            w1 = weight(c1, c2)
            w2 = weight(c2, c1)

            gain = info_gain(H, h1, h2, w1, w2)

            print(f"\nTrying split: Feature = {feature_names[f]} == {val}")
            print("Info Gain:", gain)

            if gain > MAX:
                MAX = gain
                best_feature = f
                best_value = val
                best_LX, best_RX = X[left_mask], X[right_mask]
                best_Ly, best_Ry = c1, c2

                print(">>> BEST SPLIT UPDATED")

    print("\nSELECTED SPLIT:")
    print("Feature:", feature_names[best_feature])
    print("Value:", best_value)

    return best_LX, best_RX, best_Ly, best_Ry, best_feature, best_value


def check(X, y, depth):

    if len(set(y)) == 1:
        print("PURE NODE FOUND:", y)
        return y[0]

    if depth == 0:
        print("MAX DEPTH REACHED")
        return max(set(y), key=list(y).count)

    LX, RX, Ly, Ry, f_index, f_value = nodecal(X, y)

    print("\n--- SPLIT DONE ---")
    print("LEFT:", Ly)
    print("RIGHT:", Ry)

    left_subtree = check(LX, Ly, depth - 1)
    right_subtree = check(RX, Ry, depth - 1)

    return {
        "feature": f_index,
        "value": f_value,
        "left": left_subtree,
        "right": right_subtree
    }


depth = 3
result = check(X, y, depth)


def predict(tree, x):

    if isinstance(tree, str):
        return tree

    f = tree["feature"]

    if x[f] == tree["value"]:
        return predict(tree["left"], x)
    else:
        return predict(tree["right"], x)


test_animal = {
    "EarShape": "Pointy",
    "FaceShape": "Round",
    "Whiskers": "Yes"
}

test_df = pd.DataFrame([test_animal])
test_encoded = pd.get_dummies(test_df)
test_encoded = test_encoded.reindex(columns=X_df.columns, fill_value=0)

x_test = test_encoded.values[0]

prediction = predict(result, x_test)

print("\nFINAL PREDICTION:", prediction)