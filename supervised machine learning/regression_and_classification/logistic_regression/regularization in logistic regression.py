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
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
y = np.reshape(y, (-1, 1))

def decision_boundary(w,x,b):
    return np.dot(x,w)+b
def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g
def loss_function(y,w,g,lamdha):
    esp=1e-8
    m=y.shape[0]
    L= -np.mean(y*np.log(g+esp)+(1-y)*np.log(1-g+esp))
    regularization=((lamdha/(2*m))*np.sum(w**2))
    return L+regularization
def DW(g, y, x,lamdha,w):
    m = x.shape[0]
    return (1/m) * np.dot(x.T, (g - y))+((lamdha/m)*w)
def DB(g, y):
    return np.mean(g - y)
w=np.zeros((2,1))
b=0
alpha=0.01
lamdha=2
all_loss=[]
for i in range(1000):
    z=decision_boundary(w,x,b)
    Y=sigmoid(z)
    all_loss.append(loss_function(y,w,Y,lamdha))
    w=w-alpha*DW(Y,y,x,lamdha,w)
    b=b-alpha*DB(Y,y)    
# Plot
print(w, b)
# Plot
y=y.ravel()
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
plt.title("loss")
plt.show()

print("Features shape:", x.shape)
print("Target shape:", y.shape)