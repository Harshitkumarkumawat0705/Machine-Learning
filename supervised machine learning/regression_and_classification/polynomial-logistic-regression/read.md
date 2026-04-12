# Polynomial Logistic Regression (From Scratch)

This project implements **Logistic Regression with Polynomial Features and Regularization** from scratch using NumPy.

It helps model **non-linear decision boundaries** for complex classification problems.

---

# Overview

Standard Logistic Regression creates a **linear decision boundary**.  
To handle more complex patterns, we transform input features into **polynomial features**.

---

# Feature Engineering (Polynomial Features)

Original features:
x₁, x₂  

Transformed into:
- x₁, x₂  
- x₁², x₂²  
- x₁ * x₂  
- x₁³, x₂³  

This allows the model to learn **curved decision boundaries**.

---

# Mathematical Model

z = X_poly W + b  

g(z) = 1 / (1 + e^(-z))  

Where:
- X_poly = transformed feature matrix  
- W = weights  
- b = bias  
- g(z) = sigmoid function  

---

# Dataset

- Generated using `make_classification`  
- 2 features (for visualization)  
- Noise added to simulate real-world scenarios  

---

# Loss Function (with Regularization)

L = - (1/m) Σ [y log(g) + (1 - y) log(1 - g)] + (λ / 2m) Σ w²  

Where:
- λ = regularization parameter  

---

# Gradient Descent

dW = (1/m) Xᵀ (g - y) + (λ/m) W  
db = (1/m) Σ (g - y)  

Update:
W = W - α * dW  
b = b - α * db  

---

# Key Concepts

## Polynomial Features
- Helps capture non-linear relationships  
- Expands feature space  

## Regularization
- Prevents overfitting  
- Penalizes large weights  

## Sigmoid Function
- Converts output into probability (0 to 1)  

---

# Visualization

- Scatter plot shows two classes  
- Decision boundary is plotted using contour (P = 0.5)  
- Shows how model separates classes non-linearly  

---

# Learning Outcomes

By completing this project, you will understand:

- How Logistic Regression can model non-linear data  
- Role of feature engineering (polynomial features)  
- Importance of regularization  
- Decision boundary visualization  

---

# Note

- High λ → simpler model (may underfit)  
- Low λ → complex model (may overfit)  
- Feature scaling is important for convergence  

---

# Tech Stack

- Python  
- NumPy  
- Matplotlib  
- scikit-learn (only for dataset generation)  