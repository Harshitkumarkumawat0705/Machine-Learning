# Anomaly Detection From Scratch using Gaussian Distribution

## Overview
This project implements **Anomaly Detection from scratch** using **Gaussian Distribution Probability Estimation** without using any machine learning library implementations.

The model identifies anomalous coffee roasting conditions based on:

- **Temperature (°C)**
- **Roasting Duration (minutes)**

Normal roasting conditions are learned from the training data, and low-probability observations are classified as anomalies.

---

## Dataset Description

A synthetic coffee roasting dataset was generated where:

- Optimal roasting temperature: **175°C – 260°C**
- Optimal roasting duration: **12 – 15 minutes**

Samples outside the optimal region are treated as anomalies.

---

## Workflow

### 1. Data Generation
Generated coffee roasting data using NumPy random sampling.

### 2. Data Preparation
Dataset split into:

- Training Set
- Cross Validation Set
- Testing Set

Training uses only normal examples.

---

### 3. Gaussian Parameter Estimation

Estimated:

- Mean (μ)
- Variance (σ²)

for each feature using:

μ = Mean(X)

σ² = Mean((X − μ)²)

---

### 4. Probability Estimation

Probability density computed using Gaussian Distribution:

P(x) = (1 / √(2πσ²)) × exp(−(x−μ)² / 2σ²)

Joint probability obtained by multiplying feature probabilities.

---

### 5. Threshold Selection

Best anomaly threshold (**ε**) selected using:

- Precision
- Recall
- F1 Score

Validation set used to determine optimal epsilon.

---

### 6. Evaluation

Model evaluated on test data and anomaly boundary visualized.

---

# Technologies Used

### Programming Language
- Python

### Libraries
- NumPy
- Matplotlib

### Concepts Applied
- Unsupervised Learning
- Anomaly Detection
- Gaussian Distribution
- Probability Density Estimation
- Threshold Optimization
- Precision / Recall
- F1 Score

---

# Project Structure

```bash
Anomaly-Detection-From-Scratch/
│── anomaly_detection.py
│── README.md
│── images/
