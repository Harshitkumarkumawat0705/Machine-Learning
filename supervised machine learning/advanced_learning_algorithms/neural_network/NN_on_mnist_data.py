import pandas as pd
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the fixed CSV
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")
print(train_df.shape)
print(test_df.shape)
train_data = train_df.values
y_train = train_data[:, 0]      
X_train = train_data[:, 1:] 
X_train = X_train / 255.0
num_classes = 10
y = np.eye(num_classes)[y_train]   
def ReLU(z):
    return np.maximum(0,z)
def ReLU_derivative(z):
    return (z > 0).astype(float)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoid_derivative(z):
    s=sigmoid(z)
    return (z>0)
def loss_function (y_pred,y):
    eps = 1e-8
    return -np.mean(np.sum(y * np.log(y_pred + eps), axis=1))
def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

layers=[784,256,128,10]
L=len(layers)-1
Activation=[ReLU,ReLU,softmax]
W,B=[],[]
lr=0.1
losses=[]
epochs=100

for i in range(L):
    W.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]))
    B.append(np.zeros((1,layers[i+1])))

def predict(X, W, B):
    A = X
    for i in range(len(W) - 1):
        Z = A @ W[i] + B[i]
        A = ReLU(Z)
    Z = A @ W[-1] + B[-1]
    Y_hat = softmax(Z)
    return np.argmax(Y_hat, axis=1)

def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return (correct / len(y_true)) * 100

for epoch in range(epochs):
    A=X_train
    chache=[]
#========FORWARD_PROPAGATION===========
    for i in range (L-1):
        Z = A @ W[i] + B[i]
        chache.append((A, Z))
        A = Activation[i](Z)

    Z = A @ W[-1] + B[-1]
    chache.append((A, Z))
    Y_hat = softmax(Z)

    loss = loss_function(Y_hat, y)
    losses.append(loss)
#===============back_propagation====================
    m = y.shape[0]
    dZ = (Y_hat - y) / m

    for l in reversed(range(L)):
        A_prev, Z = chache[l]
        dW = A_prev.T @ dZ
        dB = np.sum(dZ, axis=0, keepdims=True)

        W[l] -= lr * dW
        B[l] -= lr * dB

        if l != 0:
            dA = dZ @ W[l].T
            _, Z_prev = chache[l - 1]
            dZ = dA * ReLU_derivative(Z_prev)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
  

        y_train_pred = predict(X_train, W, B)
        train_accuracy = accuracy(y_train, y_train_pred)
        print(f"Training Accuracy: {train_accuracy:.2f}%")

        test_data = test_df.values
        y_test = test_data[:, 0]
        X_test = test_data[:, 1:] / 255.0

        y_test_pred = predict(X_test, W, B)
        test_accuracy = accuracy(y_test, y_test_pred)
        print(f"Test Accuracy: {test_accuracy:.2f}%")

    
# One example per class
examples = train_df.groupby("label").first().reset_index()
# Plot
plt.figure(figsize=(10, 4))
for i in range(10):
    ax = plt.subplot(2, 5, i + 1)
    img = examples.loc[i].drop("label").values.astype(np.uint8).reshape(28, 28)
    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {examples.loc[i, 'label']}")
    plt.axis("off")

plt.tight_layout()
plt.show()