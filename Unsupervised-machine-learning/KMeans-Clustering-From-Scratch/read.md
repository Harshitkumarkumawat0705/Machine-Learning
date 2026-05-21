# K-Means Clustering From Scratch

## Overview
This project implements the K-Means Clustering algorithm from scratch using NumPy without using any built-in machine learning clustering libraries.

It groups data points into clusters using:
- Random centroid initialization
- Distance computation
- Cluster assignment
- Centroid update
- Cost minimization over iterations

---

## Dataset Description
A synthetic dataset is generated using `make_classification` for visualization purposes.

- Samples: 200
- Features: 2
- Classes: 2
- Class separation: 0.8
- Noise: 10% label flipping

Note: Labels are only used for visualization. The clustering process is completely unsupervised.

---

## Algorithm Steps

### 1. Data Generation
Synthetic dataset created using sklearn's `make_classification`.

---

### 2. Initialize Centroids
Randomly select K data points as initial centroids.

K = 2

---

### 3. Compute Distance
Euclidean distance is calculated between each data point and centroid.

---

### 4. Assign Clusters
Each point is assigned to the nearest centroid.

---

### 5. Update Centroids
New centroids are computed as the mean of assigned cluster points.

---

### 6. Cost Calculation
Cost (Inertia) is computed as:

J = Σ ||x - μ||²

It is stored for each iteration to observe convergence.

---

## Tech Stack
- Python
- NumPy
- Matplotlib
- Scikit-learn (only for dataset generation)

---

## Visualizations
- Data distribution plot
- Cluster assignment with centroids
- Cost (inertia) vs iterations plot

---

## Results
- Implemented K-Means from scratch
- Observed cluster formation visually
- Tracked cost reduction over iterations
- Verified convergence behavior

---

## Project Structure
KMeans-Clustering-From-Scratch/
│── kmeans.py
│── README.md
│── images/