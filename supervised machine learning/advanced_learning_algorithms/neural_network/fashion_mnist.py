# %%
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.datasets import make_classification

# %%
df = pd.read_csv("fashion-mnist.csv")
print(df.shape)
data = df.values
np.random.shuffle(data)
print(data.shape)

# %%
shift =int(len(data)*0.8)
X_train = data[:shift]
X_test = data[shift:]
y_train = X_train[:,0]
X_train = X_train[:,1:]/255.0
print(X_train.shape)
print(y_train.shape)
y_test = X_test[:,0]
x_test = X_test[:,1:]/255.0
print(x_test.shape)
print(y_test.shape)

# %%
#one hot encoding:
y_train= np.eye(10)[y_train]
print(y_train.shape)
Layers=[784,512,256,128,10]

# %%
#Functions
def relu(z): return np.maximum(0,z)

def softmax(z):
    z-=np.max(z,axis = 1 , keepdims= True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
def Loss(Y_pred,y_train):
    esp = 1e-8
    return -np.mean(y_train*np.log(Y_pred+esp) + (1-y_train)*np.log(1-Y_pred)+esp)
def relu_derivative(z):
    return (z > 0).astype(float)
def predict(X, W, B):
    A = X
    for i in range(len(W) - 1):
        Z = A @ W[i] + B[i]
        A = relu(Z)
    Z = A @ W[-1] + B[-1]
    Y_hat = softmax(Z)
    return np.argmax(Y_hat, axis=1)

def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return (correct / len(y_true)) * 100

# %%
epochs = 50
W = []
B = []
losses = []
lr = 0.001

for i in range(len(Layers) - 1):
    W.append(np.random.randn(Layers[i], Layers[i + 1]) * np.sqrt(2 / Layers[i]))
    B.append(np.zeros((1, Layers[i + 1])))

for epoch in range(epochs):
    A = X_train
    chache = []
    #==================forwardprop======================================
    for i in range(len(Layers) - 2):
        # print(f'{i,":"}{W[i].shape} {A.shape}')
        Z = A @ W[i] + B[i]
        chache.append((A, Z))
        A = relu(Z)

    # print(f'{W[-1].shape} {A.shape}')
    Z = A @ W[-1] + B[-1]
    chache.append((A, Z))
    Y_hat = softmax(Z)

    loss = Loss(Y_hat, y_train)
    losses.append(loss)

    m = len(y_train)
    DZ = (Y_hat - y_train) / m

    #======================================backprop======================
    for i in reversed(range(len(Layers) - 1)):
        A_prev, Z = chache[i]
        # print(f'{A_prev.shape} {Z.shape}')

        DW = A_prev.T @ DZ
        DB = np.sum(DZ, axis=1, keepdims=True)

        W[i] = W[i] - lr * DW
        B[i] = B[i] - lr * DB

        # print("W shape: ", W[i].shape)

        if i != 0:
            DA = DZ @ W[i].T
            _, Z_prev = chache[i - 1]
            # print("Z previous: ", Z_prev.shape)
            DZ = A_prev * relu_derivative(Z_prev)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# %%
y_train_pred = predict(X_train, W, B)
train_accuracy = accuracy(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}%")

y_test_pred = predict(X_test, W, B)
test_accuracy = accuracy(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}%")



