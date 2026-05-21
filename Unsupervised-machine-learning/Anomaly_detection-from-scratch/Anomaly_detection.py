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

    return (X, Y)
X, y = load_coffee_data()
print(X.shape, y.shape)
X_normal = X[y == 1]
X_anomalous = X[y == 0]
rng = np.random.default_rng(42)
rng.shuffle(X_normal)
rng.shuffle(X_anomalous)

train_end = int(0.6 * len(X_normal))
cv_end = int(0.8 * len(X_normal))

X_train = X_normal[:train_end]

X_val = np.concatenate([X_normal[train_end:cv_end], X_anomalous[:len(X_anomalous)//2]])
y_val = np.concatenate([np.ones(cv_end - train_end), np.zeros(len(X_anomalous)//2)])

X_test = np.concatenate([X_normal[cv_end:], X_anomalous[len(X_anomalous)//2:]])
y_test = np.concatenate([np.ones(len(X_normal) - cv_end), np.zeros(len(X_anomalous) - (len(X_anomalous)//2))])

def estimation_gussian(x):
    m, n = x.shape
    mu = np.mean(x, axis=0)
    var = np.mean((x - mu)**2, axis = 0)
    return mu, var

def predict_gaussian(X, mu, var):
    p = (1/np.sqrt(2* np.pi * var))*np.exp(-np.square(X-mu)/(2*var))
    p_total = np.prod(p,axis=1)

    return p_total
def select_threshold(y_val, p_val):
    best_epsilon = 0
    best_f1 = 0

    step_size = (max(p_val)- min(p_val))/1000 

    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        
        predictions = (p_val < epsilon)
        
        # True Positives: Predicted anomaly (1) and actually is anomaly (y=0)
        tp = np.sum((predictions == 1) & (y_val == 0))
        # False Positives: Predicted anomaly (1) but actually normal (y=1)
        fp = np.sum((predictions == 1) & (y_val == 1))
        # False Negatives: Predicted normal (0) but actually anomaly (y=0)
        fn = np.sum((predictions == 0) & (y_val == 0))
        
        # Calculate Precision and Recall
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1-score
        if (prec + rec) > 0:
            f1 = (2 * prec * rec) / (prec + rec)
        else:
            f1 = 0
            
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
            
    return best_epsilon, best_f1

mu, var = estimation_gussian(X_train)
p_val = predict_gaussian(X_val,mu,var)
epsilon, best_f1 = select_threshold(y_val, p_val)

p_test = predict_gaussian(X_test, mu, var)
test_predictions = (p_test < epsilon)
final_preds_formatted = np.where(test_predictions, 0, 1)
accuracy = np.mean(final_preds_formatted == y_test)
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
print(f"Best Epsilon found: {epsilon}")
# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X[y.flatten() == 0][:, 0], X[y.flatten() == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(X[y.flatten() == 1][:, 0], X[y.flatten() == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.title("Dummy Binary Classification Data (for Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
# 1. First, make sure you have the epsilon value
epsilon, best_f1 = select_threshold(y_val, p_val)

# 2. Create the plot
plt.figure(figsize=(8, 6))

# Plot the original data points (using the full dataset X, y)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="red", label="Anomalies (Class 0)", alpha=0.6)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue", label="Normal (Class 1)", alpha=0.6)

# 3. Create a grid to draw the boundary line
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Calculate probabilities for every point on the grid to find the boundary
grid_points = np.c_[xx.ravel(), yy.ravel()]
p_grid = predict_gaussian(grid_points, mu, var)
p_grid = p_grid.reshape(xx.shape)

# 4. Draw the boundary line (where probability == epsilon)
plt.contour(xx, yy, p_grid, levels=[epsilon], colors='black', linewidths=3)

plt.title(f"Anomaly Detection (F1 Score: {best_f1:.2f})")
plt.xlabel("Temperature (Celsius)")
plt.ylabel("Duration (Minutes)")
plt.legend()
plt.grid(True)
plt.show()