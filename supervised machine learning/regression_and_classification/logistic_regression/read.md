# Logistic Regression Implementation (From Scratch)

This project demonstrates the implementation of Logistic Regression using Gradient Descent from scratch using NumPy.

It includes:

## Logistic Regression (Binary Classification)

## Logistic Regression with Regularization

---

# 1. Logistic Regression (Binary Classification)

## Overview
Logistic Regression is used for **classification problems**, where the output is either 0 or 1.

---

## Mathematical Model

z = w · x + b  

g(z) = 1 / (1 + e^(-z))  

Where:
- w = weights  
- b = bias  
- g(z) = sigmoid function (maps values between 0 and 1)

---

## Dataset Generation
- Synthetic dataset created using `make_classification`  
- 2 features (for visualization)  
- Noise added to simulate real-world data  

---

## Implementation Steps

### Decision Boundary
z = XW + b  

### Sigmoid Function
g(z) = 1 / (1 + e^(-z))  

Converts output into probability.

---

### Loss Function (Binary Cross-Entropy)
L = - (1/m) Σ [y log(g) + (1 - y) log(1 - g)]

- Measures classification error  
- Goal: minimize this loss  

---

### Gradient Descent

dW = (1/m) Xᵀ (g - y)  
db = (1/m) Σ (g - y)  

Update:
W = W - α * dW  
b = b - α * db  

Where:
- α = learning rate  

---

## Key Points
- Used for binary classification  
- Output is probability (0 to 1)  
- Decision boundary separates classes  

---

# 2. Logistic Regression with Regularization

## Overview
Regularization is used to **prevent overfitting** by penalizing large weights.

---

## Modified Loss Function

L = - (1/m) Σ [y log(g) + (1 - y) log(1 - g)] + (λ / 2m) Σ w²  

Where:
- λ (lambda) = regularization parameter  

---

## Gradient Descent with Regularization

dW = (1/m) Xᵀ (g - y) + (λ/m) W  
db = (1/m) Σ (g - y)  

Update:
W = W - α * dW  
b = b - α * db  

---

## Key Points
- Helps reduce overfitting  
- Penalizes large weights  
- Improves generalization  

---

# Visualization

- Scatter plot shows classification data  
- Different colors represent different classes  
- Loss vs Iterations graph shows training progress  

---

# Learning Outcomes

By completing this project, you will understand:

- How Logistic Regression works internally  
- Difference between regression and classification  
- Sigmoid function and probability mapping  
- Binary Cross-Entropy loss  
- Role of Regularization  

---

# Note

- Learning rate (α) is important  
  - Too high → unstable training  
  - Too low → slow learning  

- Regularization parameter (λ):  
  - High → underfitting  
  - Low → overfitting  

---

# Tech Stack

- Python  
- NumPy  
- Matplotlib  
- scikit-learn (only for dataset generation)  