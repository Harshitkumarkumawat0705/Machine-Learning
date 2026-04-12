# Polynomial Linear Regression (From Scratch)

This project implements **Polynomial Linear Regression** from scratch using NumPy, without using any ML libraries.

It uses a **custom dataset loaded from a CSV file** and applies **feature engineering + gradient descent** to learn the model.

---

# Overview

Linear Regression can only model **straight-line relationships**.  
To capture **non-linear patterns**, we transform input features into polynomial terms.

---

# Dataset

- Loaded from a CSV file  
- Contains:
  - x_train → input feature  
  - y_train → target value  

---

# Feature Engineering

Original feature:
x  

Transformed into:
- x  
- x²  
- x⁴  

---

## Feature Scaling

Before training, input is normalized:

x = (x - mean) / std  

This helps:
- Faster convergence  
- Stable gradient descent  

---

# Mathematical Model

y = w₁x + w₂x² + w₃x⁴ + b  

Vector form:

y = XW + b  

---

# Cost Function (Mean Squared Error)

J = (1/m) Σ (y_pred - y)²  

- Measures prediction error  
- Goal: minimize this cost  

---

# Gradient Descent

dW = (1/m) Xᵀ (y_pred - y)  
db = (1/m) Σ (y_pred - y)  

Update:
W = W - α * dW  
b = b - α * db  

Where:
- α = learning rate  

---

# Implementation Details

- Data loaded using Pandas  
- Converted to NumPy arrays  
- Polynomial features manually created  
- Gradient Descent implemented from scratch  

---

# Visualization

- Scatter plot → original data  
- Line plot → model predictions  

Shows how polynomial regression fits a **curved relationship**.

---

# Learning Outcomes

By completing this project, you will understand:

- Difference between linear and polynomial regression  
- Feature engineering for non-linear data  
- Gradient Descent optimization  
- Importance of normalization  

---

# Note

- Higher degree → more complex model  
- Too complex → overfitting  
- Learning rate affects convergence  

---

# Tech Stack

- Python  
- NumPy  
- Pandas  
- Matplotlib  