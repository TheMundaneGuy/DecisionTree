# 📘 Decision Tree Classifier (Python)

A custom implementation of a Decision Tree Machine Learning model built from scratch using Python.
This project demonstrates core ML concepts such as entropy, information gain, recursive tree construction, and classification—without relying on scikit-learn’s built-in models.

# 🚀 Project Overview

This project contains an end-to-end implementation of a decision tree classifier that:

-Reads and preprocesses CSV training/testing data

-Calculates entropy and information gain

-Recursively builds a binary decision tree

-Predicts outcomes for new samples

-Evaluates model performance (accuracy, correct/wrong predictions)

It is designed as an educational ML project and a demonstration of practical understanding of how decision trees work under the hood.

# 🧠 Key Features

✔️ Pure Python implementation (no scikit-learn)

✔️ Automatic attribute/threshold selection based on information gain

✔️ Support for numerical features

✔️ Clean tree structure with a Node class

✔️ Prediction and accuracy evaluation for test datasets

✔️ Simple and modular codebase suitable for learning or extension

# 🛠️ Technologies Used

Python

Pandas

NumPy (optional depending on your environment)

Core math & recursion (no external ML libraries)

# 📊 How It Works
### 1. Data Loading

The clean() function reads CSV files, assigns column names, and converts values to numeric.

### 2. Tree Construction

The tree is built recursively using:

Entropy calculation

Information gain maximization

Threshold-based splitting

Left/right node creation

### 3. Prediction

Each row travels down the tree until reaching a leaf node representing class 0 or 1.

### 4. Model Evaluation

The script prints:

Number of correct predictions

Number of incorrect predictions

Final accuracy percentage

# ▶️ Running the Project

Place your training and testing CSV files in the project directory and run:

```python DecisionTree.py```


The script will automatically:

-Train the decision tree using the training set

-Test it on the test set

-Display final accuracy metrics



