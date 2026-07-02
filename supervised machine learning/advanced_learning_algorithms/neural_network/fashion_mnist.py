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
X_train = (X_train[:,1:]/255.0-0.5)*2
print(X_train.shape)
print(y_train.shape)
y_test = X_test[:,0]
x_test = (X_test[:,1:]/255.0 - 0.5)*2
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
def Loss(Y_pred, y_true):
    eps = 1e-8
    return -np.mean(np.sum(y_true * np.log(Y_pred + eps), axis=1))
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
Batch_size= 64

for i in range(len(Layers) - 1):
    W.append(np.random.randn(Layers[i], Layers[i + 1]) * np.sqrt(2 / Layers[i]))
    B.append(np.zeros((1, Layers[i + 1])))

for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    for l in range (0,len(X_train),Batch_size):
        X_batch = X_train[l:l+Batch_size]
        y_batch = y_train[l:l+Batch_size]
        A = X_batch
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

        loss = Loss(Y_hat, y_batch)
        losses.append(loss)

        m = len(y_batch)
        DZ = (Y_hat - y_batch) / m

        #===============backprop======================
        for j in reversed(range(len(Layers) - 1)):
            A_prev, Z = chache[j]
            # print(f'{A_prev.shape} {Z.shape}')

            DW = A_prev.T @ DZ
            DB = np.sum(DZ, axis=0, keepdims=True)
            lamdha = 0.0005

            W[j] = W[j] - lr * (DW+ lamdha*W[j])
            B[j] = B[j] - lr * DB

            # print("W shape: ", W[i].shape)

            if j != 0:
                DA = DZ @ W[j].T
                _, Z_prev = chache[j - 1]
                # print("Z previous: ", Z_prev.shape)
                DZ = DA * relu_derivative(Z_prev)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# %%
y_train_labels = np.argmax(y_train, axis=1)

y_train_pred = predict(X_train, W, B)
train_accuracy = accuracy(y_train_labels, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}%")

y_test_pred = predict(x_test, W, B)
test_accuracy = accuracy(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}%")


