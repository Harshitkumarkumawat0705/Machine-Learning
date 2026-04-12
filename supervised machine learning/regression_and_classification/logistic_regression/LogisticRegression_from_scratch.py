import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
# Set seed
np.random.seed(42)
# Generate a classification dataset
x, y = make_classification(
    n_samples=200,
    n_features=2,          # Only 2 useful features for visualization
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,         # Low separation makes it harder
    flip_y=0.1,            # Add noise (10% label flipping)
    random_state=42
)
def decision_boundary(w,x,b):
    return np.dot(x,w)+b
def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g
def cost_function(y,x,g):
    L= np.mean(-y*np.log(g)-(1-y)*np.log(1-g))
    return L
def DW(g, y, x):
    return  np.dot(x.T, (g - y))/x.shape[0]
def DB(g, y):
    return np.sum(g - y)/x.shape[0]
w=np.zeros((2,1))
b=0
alpha=0.001
all_loss=[]
for i in range(1000):
    z=decision_boundary(w,x,b)
    Y=sigmoid(z)
    all_loss.append(cost_function(y,x,Y))
    w=w-alpha*DW(Y,y,x)
    b=b-alpha*DB(Y,y)    
# Plot
print(w, b)
plt.figure(figsize=(8, 6))
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.title("Dummy Binary Classification Data (for Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
plt.plot(all_loss)
plt.show()

print("Features shape:", x.shape)
print("Target shape:", y.shape)

x = (x - (np.mean(x))) / np.std(x)
y_reshaped = np.reshape(y, (-1, 1))