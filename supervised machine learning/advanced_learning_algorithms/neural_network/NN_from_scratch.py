import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Set seed
np.random.seed(42)

# Generate a classification dataset
X, y = make_classification(
    n_samples=200,
    n_features=2,          # Only 2 useful features for visualization
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,         # Low separation makes it harder
    flip_y=0.1,            # Add noise (10% label flipping)
    random_state=42
)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
y = np.reshape(y, (-1, 1))
def ReLU(z):
    return np.maximum(0,z)
def ReLU_derivative(z):
    return (z > 0).astype(float)
def sigmoid(z):
    return 1/(1+np.exp(-z))  
def sigmoid_derivative(z):
    s=sigmoid(-z)
    return s*(1-s)
def loss_fn (Y_pred,W,y,lamdha):
    esp=1e-8
    m=y.shape[0]
    L= -np.mean(y*np.log(Y_pred + esp)+(1-y)*np.log(1-Y_pred +esp))
    regularization = 0
    for i in range (len(W)):
        regularization=regularization+(lamdha / (2 * m)) * np.sum(W[i] ** 2)
    return L+regularization
def forward_pass(X, W, B):
    A = X
    L = len(W)

    for i in range(L - 1):
        Z = A @ W[i] + B[i]
        A = ReLU(Z)

    Z = A @ W[-1] + B[-1]
    Y_hat = sigmoid(Z)
    return Y_hat 
layers =[2,512,128,1]
activation = [ReLU, ReLU, sigmoid]
L=len(layers)-1
W, B = [], []
lr=0.01
lamdha = 0
epochs=2000
losses = []
for i in range(L):
    W.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]))
    B.append(np.zeros((1, layers[i+1])))
for epoch in range(epochs):
    A=X
    chache =[]

#==========Forward propagation===========
    for i in range (L-1):
        Z = np.dot(A,W[i])+B[i]
        chache.append({
            "A_prev": A,"Z": Z,"fn":ReLU
        })
        A = ReLU(Z)
    Z=np.dot(A,W[-1])+B[-1]
    chache.append({
            "A_prev": A,"Z": Z
        })
    Y_hat=sigmoid(Z)
    loss = loss_fn(Y_hat,W,y,lamdha)
    losses.append(loss)

#=============back_propagation============
    m=y.shape[0]
    dZ = (Y_hat-y) / m

    for l in reversed(range(L)):
        A_prev = chache[l]["A_prev"]
        Z = chache[l]["Z"]

        if l != L-1:
            dZ = dA * ReLU_derivative(Z)
        dW = (A_prev.T @ dZ) + (lamdha/m) * W[l]
        db = np.sum(dZ, axis=0, keepdims=True)
        W[l] -= lr * dW
        B[l] -= lr * db
        dA = dZ @ W[l].T
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
# ===== Decision Boundary =====

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]  
Z_pred = forward_pass(grid, W, B)
Z_pred = Z_pred.reshape(xx.shape)
#loss visualization
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()   

# Plot
y_flat = y.ravel()
plt.figure(figsize=(8, 6))

# ===== Decision boundary =====
plt.contourf(xx, yy, Z_pred > 0.5, alpha=0.3)

# ===== Data points =====
plt.scatter(X[y_flat == 0][:, 0], X[y_flat == 0][:, 1],
            color="red", label="Class 0", edgecolor="k")

plt.scatter(X[y_flat == 1][:, 0], X[y_flat == 1][:, 1],
            color="blue", label="Class 1", edgecolor="k")

plt.title("Decision Boundary (Neural Network)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()


