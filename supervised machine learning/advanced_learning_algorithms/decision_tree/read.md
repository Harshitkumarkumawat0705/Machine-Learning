# Decision Tree & Information Gain from Scratch (NumPy + Pandas)

This project contains **multiple implementations of Decision Trees built completely from scratch** without using sklearn.

The goal is to deeply understand:
- How decision trees choose splits using **Entropy & Information Gain**
- How categorical features are handled
- How trees grow recursively
- How decision boundaries are formed visually

---

# 1. Decision Tree using Information Gain (Basic Categorical Model)

## Overview

This model builds a decision tree using a simple dataset (`animal_features.csv`) containing categorical features like:

- EarShape  
- FaceShape  
- Whiskers  

The goal is to classify animals (Cat / Dog etc.) using **entropy-based splitting**.

---

## Key Concepts Used

### 1. Entropy
Measures impurity in a node:

H = -p log2(p) - (1-p) log2(1-p)

Lower entropy → purer node

---

### 2. Information Gain
Measures how good a split is:

IG = H(parent) - [W1 * H(left) + W2 * H(right)]

Higher IG → better split

---

## How the Tree is Built

### Step 1: Data Encoding
- Categorical features are converted using `pd.get_dummies()`

---

### Step 2: Best Split Selection
For every feature and value:
- Split dataset into left & right groups  
- Compute entropy for both groups  
- Calculate Information Gain  
- Select split with maximum gain  

---

### Step 3: Recursive Tree Building
Function `check()`:
- Stops if node is pure  
- Stops if max depth is reached  
- Otherwise keeps splitting recursively  

---

## Prediction Flow

- Traverse tree based on feature condition  
- Move left or right depending on value match  
- Return final class label  

---

## What You Learn
- How decision trees choose splits manually  
- Role of entropy in classification  
- Recursive tree construction logic  
- Prediction using tree traversal  

---

# 2. Extended Decision Tree (Multi-Feature Categorical Model)

## Overview

This is an advanced version of the decision tree using a larger dataset (`animal_features_extended.csv`).

It includes multiple categorical features:

- EarShape  
- FaceShape  
- Whiskers  
- FurLength  
- BodySize  
- Tail  
- DietType  

---

## Key Improvements Over Previous Model

### 1. More Features
- Handles multiple categorical inputs  
- Higher dimensional decision making

---

### 2. Index-Based Tree Building
Instead of working directly with values:
- Uses row indices  
- Splits dataset using index filtering  

---

### 3. Pure Recursive Splitting
At each node:
- Compute entropy of current labels  
- Try all features  
- Choose best feature based on Information Gain  

---

## Tree Construction Process

### Step 1: Encode Data
- One-hot encoding using `pd.get_dummies()`

---

### Step 2: Feature Evaluation
For each feature:
- Split into left/right groups  
- Compute entropy  
- Calculate Information Gain  
- Select best feature  

---

### Step 3: Recursive Splitting
Function `check()`:
- Stops when pure node is found  
- Stops at max depth  
- Otherwise splits further  

---

## Prediction Flow

- Traverse tree using feature index  
- Follow left/right branches  
- Return majority class at leaf  

---

## What You Learn
- Multi-feature decision tree logic  
- Index-based dataset splitting  
- Importance of feature selection  
- How categorical trees scale with data  

---

# 3. Decision Boundary using Information Gain (Continuous Split Tree)

## Overview

This project builds a decision tree for a **continuous 2D dataset (Coffee Roasting Problem)** and visualizes the **decision boundary**.

Dataset is generated based on:

- Temperature  
- Roasting Time  

---

## Key Idea

The model learns:
- Whether coffee is properly roasted or not  
based on a **non-linear rule-based system**

---

## Tree Building Logic

### 1. Entropy Calculation
Same entropy formula used for impurity measurement.

---

### 2. Information Gain
Used to choose best split:

IG = H - (W1 * H1 + W2 * H2)

---

### 3. Continuous Splitting
Unlike categorical trees:
- Each feature value is treated as a possible threshold  
- Splits are made using:

X > threshold

---

## Tree Construction

Function `build_tree()`:
- Stops if:
  - All labels are same  
  - Max depth reached  
- Otherwise:
  - Finds best feature + threshold  
  - Splits dataset  
  - Builds left and right subtree  

---

## Prediction

Function `predict_one()`:
- Traverse tree recursively  
- Compare feature value with threshold  
- Move left or right  

---

## Decision Boundary Visualization

- Mesh grid is created over feature space  
- Each point is predicted using the tree  
- Contour plot shows classification regions  

This clearly shows:
- How tree splits space into rectangular regions  
- How decision boundaries are piecewise linear  

---

## What You Learn
- Continuous feature splitting in decision trees  
- Threshold-based learning  
- How trees create decision boundaries  
- Visualization of classification regions  

---

# Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  

---

# Overall Learnings from All Projects

- Entropy & Information Gain from scratch  
- Decision tree construction (categorical + continuous)  
- Recursive tree logic  
- Feature selection process  
- Decision boundary visualization  
- How simple rules build complex models  

---