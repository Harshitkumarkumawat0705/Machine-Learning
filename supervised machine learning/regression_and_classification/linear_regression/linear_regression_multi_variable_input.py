import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)

# Number of samples and features
num_samples = 100
num_features = 3

# Generate random x values (features)
x_train = 2 * np.random.rand(num_samples, num_features)

# Define true weights (slopes) and intercept
true_weights = np.array([[3], [2], [1]])  # shape: (3, 1)
true_intercept = 4

# Generate noise
noise = np.random.randn(num_samples, 1)

# Compute y values
y_train = true_intercept + x_train @ true_weights + noise  # @ is matrix multiplication

# Plotting using only first feature for visualization
plt.figure(figsize = (8, 5))
plt.scatter(x_train[:, 0], y_train, color = "green", label = "Training data (vs 1st feature)", alpha = 0.7)
plt.title("Dummy Linear Regression Data (3 features)")
plt.xlabel("x_train[:, 0] (1st feature)")
plt.ylabel("y_train")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape is:", x_train.shape)
print("Target shape is:", y_train.shape)


def linear_regresssion(W, b, x_tr):    
    y_pred = np.dot(x_tr, W) + b
    return y_pred

def cost_function(Y, y):
    cost=(Y-y)**2
    return np.mean(cost)

def DW(Y, y,x):
    m = x.shape[0]
    k=x.reshape(3,100)
    sum = np.dot(k,(Y-y))
    return sum/m

def DB(Y, y):
    m = Y.shape[0]
    total = 0
    for i in range(m):
        total += (Y[i] - y[i])
    return total / m

W = np.array([[1],[3],[4]])
b=0
alpha = 0.001
all_cost=[]

#gradient decent
for i in range (1000):
    y_pred=linear_regresssion(W,b,x_train)
    cost = cost_function(y_pred,y_train)
    all_cost.append(cost)
    W=  W-(alpha*DW(y_pred,y_train,x_train))  
    b= b-(alpha*DB(y_pred,y_train))

plt.plot(all_cost)
plt.show()

print(W, b)
    