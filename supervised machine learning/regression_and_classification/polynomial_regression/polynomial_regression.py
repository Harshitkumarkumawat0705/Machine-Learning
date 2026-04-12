import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

csv_file = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\New folder (2)\polynomial_data.csv")

print(csv_file)
cx_train=csv_file["x_train"]
x1 = cx_train.to_numpy()
#NORMALIZAtION OF X1
mean=np.mean(x1)
stndd=np.std(x1)
print("standard_daviation: ",stndd)
new_xtrain = (x1-mean)/stndd
x2 = new_xtrain**2
x3 = new_xtrain**4
X_train= np.hstack((new_xtrain.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1)))
print(X_train.shape)

cx_train=csv_file["y_train"]

y1 = cx_train.to_numpy()
Y_train = y1.reshape(140,1)

plt.scatter(new_xtrain,y1,color="green")
plt.show()

def linear_regresssion(W, b, x_tr):    
    y_pred = np.dot(x_tr, W) + b
    return y_pred

def cost_fn(Y,y):
     cost=(Y-y)**2
     return np.mean(cost)
def DW(Y,y,x):
    m = x.shape[0]
    k=x.T
    sum = np.dot(k,(Y-y))
    return sum/m
def DB(Y,y):
    m = Y.shape[0]
    return np.sum(Y-y) / m


W = np.array([[.1],[.3],[.4]])
b=0
alpha = 0.01
all_cost=[]

#gradient decent
for i in range (10000):
    y_pred=linear_regresssion(W,b,X_train)
    
    cost = cost_fn(y_pred, Y_train)
    
    all_cost.append(cost)
    
    W=  W-(alpha*DW(y_pred,Y_train,X_train))  
    
    b= b-(alpha*DB(y_pred,Y_train))

plt.scatter(new_xtrain,y1,color="green")
plt.plot(new_xtrain,y_pred[:,0],color="red")
plt.show()

print(x1.shape, y_pred[:,0].shape)
   
    