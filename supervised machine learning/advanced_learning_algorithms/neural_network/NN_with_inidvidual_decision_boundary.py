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
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
y = np.reshape(y, (-1, 1))


#==============function_used===================
def ReLU(z):
    return np.maximum(0,z)

def ReLU_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

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

# ======important_variables=======

layers =[2,3,1]
activation = [ReLU, sigmoid]
L=len(layers)-1
W, B = [], []
lr=0.01
lamdha = 0
epochs=20000
losses = []

for i in range(L):
    W.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i+1]))
    B.append(np.zeros((1, layers[i+1])))
for epoch in range(epochs):
    A=X
    chache =[]

#==========Forward propagation===========
    for i in range (L):
        Z = np.dot(A,W[i])+B[i]
        chache.append({
            "A_prev": A,"Z": Z
        })
        A = activation[i](Z)
    loss = loss_fn(A,W,y,lamdha)
    losses.append(loss)

#=============back_propagation============
    m=y.shape[0]
    dZ = (A-y) / m

    for l in reversed(range(L)):
        A_prev = chache[l]["A_prev"]
        Z = chache[l]["Z"]

        if l != L-1:
            dZ = dA * ReLU_derivative(Z)
        dW = (A_prev.T @ dZ) + (lamdha/m) * W[l]
        db = np.sum(dZ, axis=0, keepdims=True)
        dA = dZ @ W[l].T
        W[l] -= lr * dW
        B[l] -= lr * db
    if epoch % 500 == 0:
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
plt.figure(figsize=(8, 6))
plt.scatter(X[y.flatten() == 0][:, 0], X[y.flatten() == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(X[y.flatten() == 1][:, 0], X[y.flatten() == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.title("Dummy Binary Classification Data (for Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
W1=W[0]
B1=B[0]

for i in range (3):
    w1=W1[0,i]
    w2=W1[1,i]
    b = B1[0,i]
    x_vals=np.linspace(x_min,x_max,200)
    y_vals=(-(w1*x_vals+b)/w2)
    plt.plot(x_vals, y_vals, '--', label=f'Hidden neuron {i+1}')
plt.scatter(X[y.ravel()==0][:,0], X[y.ravel()==0][:,1], c='red', edgecolor='k', label='Class 0')
plt.scatter(X[y.ravel()==1][:,0], X[y.ravel()==1][:,1], c='blue', edgecolor='k', label='Class 1')

plt.contour(xx, yy, Z_pred, levels=[0.5], colors='black', linewidths=2)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend()
plt.title("Decision Boundary + Hidden Neuron Lines")
plt.show()

