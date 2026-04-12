# Linear Regression From Scratch (NumPy)

This project implements **Linear Regression from scratch** using **NumPy**, without using any ML libraries like scikit-learn.

It includes:
- Single Variable Linear Regression
- Multiple Variable Linear Regression

---

## 1. Single Variable Linear Regression

### Overview
This model predicts the output using a single input feature.

### Mathematical Model
f(x) = w * x + b  

Where:
- w = weight (slope)  
- b = bias (intercept)

---

### Dataset Generation
- Random values of x are generated
- Target is created using:

y = 4 + 3x + noise  

This simulates real-world noisy data.

---

### Implementation

#### Prediction Function
y_pred = w * x + b  

#### Cost Function (Mean Squared Error)
J = (1 / 2m) Σ (y_pred - y)²  

- Measures error between prediction and actual values  
- Goal: minimize this cost  

---

### Gradient Descent
w = w - α * dw  
b = b - α * db  

Where:
- α = learning rate  

---

### Key Points
- Implemented using loops (beginner-friendly)  
- Helps understand internal working of ML models  
- Output is continuous  

---

## 2. Multiple Variable Linear Regression

### Overview
This model predicts output using multiple input features.

---

### Mathematical Model
f(x) = w₁x₁ + w₂x₂ + w₃x₃ + ... + b  

Vector form:
y = XW + b  

---

### Dataset Generation
- 3 input features generated  
- True weights:

[3, 2, 1]

- Target:

y = 4 + (3x₁ + 2x₂ + 1x₃) + noise  

---

### Implementation

#### Prediction (Vectorized)
y_pred = X @ W + b  

- Uses matrix multiplication (efficient)

---

### Cost Function
J = (1 / 2m) Σ (y_pred - y)²  

---

### Gradient Descent
dW = (1/m) Xᵀ (y_pred - y)  
db = (1/m) Σ (y_pred - y)  

Update:
W = W - α * dW  
b = b - α * db  

---

### Key Points
- Uses vectorized operations (fast & efficient)  
- Scales well for large datasets  
- Used in real-world ML systems  

---

## Visualization

- Scatter plot shows training data  
- Line plot shows model prediction (single variable)  
- Cost vs Iterations graph shows learning progress  

---

## Learning Outcomes

By completing this project, you will understand:

- How Linear Regression works internally  
- Difference between single & multiple variables  
- Gradient Descent optimization  
- Importance of vectorization  

---

## Note

- Learning rate (α) is very important  
  - Too high → model may diverge  
  - Too low → slow convergence  

---

## Tech Used

- Python  
- NumPy  
- Matplotlib  

---

## Author

Implemented as part of learning Machine Learning fundamentals 