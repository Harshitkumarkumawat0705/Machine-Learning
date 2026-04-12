import numpy as np
import matplotlib.pyplot as plt


# Set the seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 100

# Generate random x values (features)
x_train = 2 * np.random.rand(num_samples, 1)

# Generate corresponding y values with a linear relationship (y = 4 + 3x + noise)
true_slope = 3
true_intercept = 4
noise = np.random.randn(num_samples, 1)

y_train = true_intercept + true_slope * x_train + noise

# Plotting the data
plt.figure(figsize = (8, 5))
plt.scatter(x_train, y_train, color = "blue", label = "Training data", alpha = 0.7)
plt.title("Dummy Linear Regression Data")
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape is:", x_train.shape)
print("Target shape is:", y_train.shape)

#m = number of examples
#w = slope

def linear_regression(x, y, w, b):
    y_pred = np.zeros_like(y)
    m = x.shape[0]
    for i in range(m):
        y_pred[i][0] = w * x[i][0] + b
    return y_pred 

def cost_function(yp, y):
    cost =0
    m = y.shape[0]
    for i in range(m):
         cost = cost + ((yp[i][0]-y[i][0])**2)
    return cost / (2 * m)

def Dw(x, y, w, b):
    m = x.shape[0]
    dw = 0
    y_pred = linear_regression(x, y, w, b)
    for i in range (m):
        dw += (y_pred[i][0]-y[i][0])*x[i][0]
    return dw / m

def Db(x, y, w, b):
    m = x.shape[0]
    y_pred = linear_regression(x, y, w, b)
    db= 0
    for i in range (m):
        db += (y_pred[i][0]-y[i][0])
    return db /m

w = 2
b = 0
alpha=0.5
all_cost=[]
for i in range (10000):
    w = w - (alpha* Dw(x_train,y_train,w,b))
    b = b - (alpha* Db(x_train,y_train,w,b))
    
    y_pred = linear_regression(x_train,y_train,w,b)
    all_cost.append(cost_function(y_pred,y_train))

plt.figure(figsize = (8, 5))
plt.scatter(x_train, y_train, color = "blue", label = "Training data", alpha = 0.7)
plt.plot(x_train, y_pred, color = "red", label = "Training data", alpha = 0.7)
plt.title("Dummy Linear Regression Data")
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(all_cost)
plt.show()

print(w, b)