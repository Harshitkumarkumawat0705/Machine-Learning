import pandas as pd                                     #SAHH
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("animal_features_extended.csv") 
print(data)
X = pd.get_dummies(
    data[["EarShape", "FaceShape", "Whiskers",
          "FurLength", "BodySize", "Tail", "DietType"]]
)
new = data["Animal"].values

features = []
Features = []
for col in X.columns:
    features.append(X[col].values)
    Features.append(col)
def pro1(labels):
    values, counts = np.unique(labels, return_counts=True)
    probablities = counts / counts.sum()
    return probablities
def weight (list1,list2):
      return len(list1)/(len(list1)+len(list2))

def entropy(pbs):
    H = 0
    for p in pbs:
        if p > 0:
            H -= p * np.log2(p)
    return H


def info_gain(W1,W2,H1,H2,H):
    return H-(W1*H1 + W2*H2)

def nodecal(indixes):
    MAX = -1
    selected_feature = None
    best_left = None
    best_right = None

    # labels for current node
    labels = new[indixes]
    H = entropy(pro1(labels))

    for f_idx, feature in enumerate(features):

        left_idx = []
        right_idx = []
        for idx in indixes:
            if feature[idx] == 1:
                left_idx.append(idx)
            else:
                right_idx.append(idx)
        if len(left_idx) == 0 or len(right_idx) == 0:
            continue
        H_left = entropy(pro1(new[left_idx]))
        H_right = entropy(pro1(new[right_idx]))

        
        w_left = len(left_idx) / len(indixes)
        w_right = len(right_idx) / len(indixes)

        
        gain = info_gain(w_left, w_right, H_left, H_right, H)

        print("info_gain:", gain, "feature:", Features[f_idx])

        if gain > MAX:
            MAX = gain
            selected_feature = f_idx
            best_left = left_idx
            best_right = right_idx

    return best_left, best_right, selected_feature

print("new data: ",new)
def check(data, depth):
    labels = new[data]
    if len(set(labels)) == 1:
        print("pure class:", labels)
        return labels.tolist()

    if depth == 0:
        print("MAXIMUM DEPTH REACHED!")
        return data

    L1, L2, f_index = nodecal(data)
    print("       ")
    print("left split: ",L1)
    print("       ")
    print("Right split: ",L2)
    print("       ")
    left_subtree  = check(L1, depth-1)
    right_subtree = check(L2, depth-1)

    return {
    "feature": f_index,
    "left": left_subtree,
    "right": right_subtree
}
depth = 3
indices = np.arange(len(new))
result = check(indices, depth)
print ("left_subtree: ",result["left"])
print ("right_subtree: ",result["right"])
def majority_class(data):
    return max(set(data), key=data.count)
def predict(tree, x):
    if isinstance(tree, list):
        return max(set(tree), key=tree.count)

    f = int(tree["feature"])

    # SAME rule as training
    if x[f] == 1:
        return predict(tree["left"], x)
    else:
        return predict(tree["right"], x)

test_animal = {
    "EarShape": "Pointy",
    "FaceShape": "Round",
    "Whiskers": "Yes",
    "FurLength": "Long",
    "BodySize": "Large",
    "Tail": "Yes",
    "DietType": "Carnivore"
}

test_df = pd.DataFrame([test_animal])
test_encoded = pd.get_dummies(test_df)

test_encoded = test_encoded.reindex(columns=X.columns, fill_value=0)
x_test = test_encoded.values[0]
prediction = predict(result, x_test)
print("Prediction:", prediction)

