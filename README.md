# Anomaly Detection

---

In this project, I implemented an anomaly detection algorithm and applied it to detect failing servers in a network. The approach is based on fitting a Gaussian distribution to server performance metrics, identifying outliers, and validating performance with an \$F\_1\$ score.

The project starts with a simple 2D dataset for visualization and then extends to a high-dimensional dataset to simulate more realistic server monitoring.

---

# Outline

* [1 - Packages](#1---packages)
* [2 - Anomaly Detection](#2---anomaly-detection)

  * [2.1 Problem Statement](#21-problem-statement)
  * [2.2 Dataset](#22-dataset)
  * [2.3 Gaussian Distribution](#23-gaussian-distribution)

    * [Exercise 1 Solution](#exercise-1-solution)
    * [Exercise 2 Solution](#exercise-2-solution)
  * [2.4 High-Dimensional Dataset](#24-high-dimensional-dataset)

---

<a name="1---packages"></a>

## 1 - Packages

For this project, I used the following:

* [NumPy](https://numpy.org) for numerical computations.
* [Matplotlib](http://matplotlib.org) for visualizations.
* `utils.py` for helper functions like data loading and visualization.

```python
import numpy as np
import matplotlib.pyplot as plt
from utils import *

%matplotlib inline
```

---

<a name="2---anomaly-detection"></a>

## 2 - Anomaly Detection

<a name="21-problem-statement"></a>

### 2.1 Problem Statement

The goal was to detect anomalous behavior in server computers.

* Dataset features:

  * **Throughput (mb/s)**
  * **Latency (ms)**

I collected \$m=307\$ examples. Most examples represent normal behavior, but some are anomalies. Using a Gaussian model, I fit the data distribution, then flagged values with very low probabilities as anomalies.

---

<a name="22-dataset"></a>

### 2.2 Dataset

I loaded the dataset into `X_train`, `X_val`, and `y_val`.

* `X_train`: used to fit the Gaussian distribution.
* `X_val` and `y_val`: used for cross-validation to select a threshold \$\varepsilon\$.

Example code:

```python
X_train, X_val, y_val = load_data()

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
```

I also visualized the dataset with a scatter plot of throughput vs. latency:

```python
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b')
plt.title("Server Metrics (Throughput vs Latency)")
plt.ylabel('Throughput (mb/s)')
plt.xlabel('Latency (ms)')
plt.axis([0, 30, 0, 30])
plt.show()
```

---

<a name="23-gaussian-distribution"></a>

### 2.3 Gaussian Distribution

To model the data distribution, I estimated the mean (\$\mu\$) and variance (\$\sigma^2\$) for each feature, then used these to compute probabilities.

---

<a name="exercise-1-solution"></a>

### ✅ Exercise 1 Solution – Estimating Gaussian Parameters

**Task:** Implement `estimate_gaussian` to calculate mean and variance of each feature.

```python
def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features in the dataset.
    Args:
        X (ndarray): (m, n) Data matrix
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    m, n = X.shape
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    return mu, var

# Run it
mu, var = estimate_gaussian(X_train)

print("Mean of each feature:", mu)
print("Variance of each feature:", var)
```

**Expected Output:**

```
Mean of each feature: [14.11222578 14.99771051]
Variance of each feature: [1.83263141 1.70974533]
```

---

<a name="exercise-2-solution"></a>

### ✅ Exercise 2 Solution – Selecting the Threshold

**Task:** Implement `select_threshold` to find the best \$\varepsilon\$ based on the \$F\_1\$ score.

```python
def select_threshold(y_val, p_val): 
    best_epsilon = 0
    best_F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = (p_val < epsilon)

        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))

        if tp + fp == 0 or tp + fn == 0:
            continue

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = 2 * prec * rec / (prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    
    return best_epsilon, best_F1

# Run it
p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)

print('Best epsilon found:', epsilon)
print('Best F1 on Cross Validation Set:', F1)
```

**Expected Output:**

```
Best epsilon found: 8.99e-05
Best F1 on Cross Validation Set: 0.875
```

I then plotted the results, marking anomalies in red.

---

<a name="24-high-dimensional-dataset"></a>

### 2.4 High-Dimensional Dataset

Finally, I extended the method to a dataset with 11 features.

```python
X_train_high, X_val_high, y_val_high = load_data_multi()

mu_high, var_high = estimate_gaussian(X_train_high)
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)

epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('Best epsilon (high-dim):', epsilon_high)
print('Best F1 (high-dim):', F1_high)
print('# Anomalies found:', np.sum(p_high < epsilon_high))
```

**Expected Output:**

```
Best epsilon (high-dim): 1.38e-18
Best F1 (high-dim): 0.615385
# Anomalies found: 117
```

---

# ✅ Summary

* Implemented anomaly detection using a Gaussian model.
* Estimated mean & variance of features (`estimate_gaussian`).
* Used cross-validation to find the best threshold (`select_threshold`).
* Successfully identified anomalies in both 2D and high-dimensional datasets.

This project demonstrates how anomaly detection can be used in server monitoring to flag unusual behavior before failures occur.

---
