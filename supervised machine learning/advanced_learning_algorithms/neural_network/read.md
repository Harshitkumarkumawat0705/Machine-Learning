# Neural Network from Scratch (NumPy)

This project contains **three implementations of Neural Networks built completely from scratch using NumPy**.

The goal is to deeply understand how neural networks work internally — including forward propagation, backpropagation, loss computation, and decision boundaries — without using deep learning frameworks like TensorFlow or PyTorch.

---

# 1. Dummy Classification Neural Network

## Overview
A synthetic dataset is generated using `sklearn.datasets.make_classification`.

The dataset is intentionally made more realistic by:
- Low class separation
- 10% label noise (`flip_y=0.1`)

This makes the learning problem harder and helps understand how neural networks behave on noisy data.

---

## Model Architecture

- Input Layer: 2 features  
- Hidden Layer 1: 512 neurons (ReLU)  
- Hidden Layer 2: 128 neurons (ReLU)  
- Output Layer: 1 neuron (Sigmoid)

---

## Key Steps Explained

### Data Generation
- `make_classification()` is used to generate a binary dataset
- Noise is added using `flip_y=0.1`
- Data is standardized using z-score normalization:
  
X = (X - mean) / std

---

### Forward Propagation
Each layer performs:

Z = XW + b  
A = activation(Z)

- ReLU is used in hidden layers → handles non-linearity  
- Sigmoid is used in output layer → outputs probability (0–1)

---

### Loss Function
Binary Cross-Entropy Loss is used:

Loss = -[y log(ŷ) + (1-y) log(1-ŷ)]

Regularization is included to prevent overfitting.

---

### Backpropagation
- Gradients are calculated manually using chain rule  
- Errors are propagated backward layer by layer  
- Parameters are updated using Gradient Descent:

W = W - α * dW  
b = b - α * db

---

### Visualization
- Training loss curve is plotted  
- Decision boundary is visualized using a mesh grid  
- Shows how the model separates two classes in 2D space  

---

## What You Learn
- How neural networks learn non-linear boundaries  
- Effect of noisy data on learning  
- Manual implementation of forward and backward propagation  
- Decision boundary visualization  

---

# 2. MNIST Digit Classification Neural Network

## Overview
This project classifies handwritten digits (0–9) using the MNIST dataset.

Each image:
- 28 × 28 pixels  
- Flattened into 784 features  
- Normalized to range [0, 1]

---

## Model Architecture

- Input Layer: 784 neurons  
- Hidden Layer 1: 256 neurons (ReLU)  
- Hidden Layer 2: 128 neurons (ReLU)  
- Output Layer: 10 neurons (Softmax)

---

## Key Steps Explained

### Data Preprocessing
- Labels are one-hot encoded  
- Pixel values are normalized by dividing by 255  

---

### Forward Propagation
- ReLU activation in hidden layers  
- Softmax activation in output layer  

Softmax converts outputs into probability distribution across 10 classes.

---

### Loss Function
Categorical Cross-Entropy Loss:

Loss = -Σ y * log(ŷ)

---

### Backpropagation
- Error starts from Softmax output  
- Gradients flow backward through layers  
- ReLU derivative is applied in hidden layers  
- Weights updated using Gradient Descent  

---

### Evaluation
- Training accuracy is calculated  
- Test accuracy is evaluated on unseen data  
- Performance is tracked over epochs  

---

### Visualization
- Sample MNIST digits are displayed  
- Loss curve over training is plotted  

---

## What You Learn
- Multi-class classification using neural networks  
- Softmax + Cross-Entropy combination  
- Working with image data using NumPy  
- Train vs test evaluation  

---

# 3. Coffee Roasting Classification Neural Network

## Overview
This project classifies coffee as properly roasted or not based on:
- Temperature  
- Roasting time  

The dataset is manually generated using logical rules, making it a non-linear classification problem.

---

## Model Architecture

- Input Layer: 2 features  
- Hidden Layer: 3 neurons (ReLU)  
- Output Layer: 1 neuron (Sigmoid)

---

## Key Steps Explained

### Custom Dataset Creation
- Random values are generated for temperature and time  
- Labels are assigned using conditions:
  - Correct roasting range → Class 1  
  - Otherwise → Class 0  

This creates a non-linear decision boundary.

---

### Feature Scaling
X = (X - mean) / std

Helps gradient descent converge faster.

---

### Forward Propagation
- Hidden layer learns intermediate representations  
- Output layer gives probability using Sigmoid  

---

### Backpropagation
- Gradients computed manually using chain rule  
- Model learns decision boundaries from scratch  

---

### Decision Boundary Analysis
- Mesh grid used to visualize full feature space  
- Final decision boundary plotted  
- Each hidden neuron’s individual linear boundary is also shown  

This demonstrates how:
- Each neuron learns a simple line  
- Combined layers create complex decision boundaries  

---

## Output
- Dataset visualization  
- Loss curve  
- Final decision boundary  
- Hidden neuron boundary lines  

---

## What You Learn
- Non-linear classification from simple rules  
- Role of hidden neurons  
- Geometric interpretation of neural networks  
- How multiple linear boundaries combine into complex ones  

---

# Technologies Used

- Python  
- NumPy  
- Matplotlib  
- Scikit-learn (only for dataset generation in Project 1 and MNIST loading)