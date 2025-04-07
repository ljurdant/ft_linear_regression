# ft_linear_regression

# 🚗 Simple Linear Regression: Car Price Predictor

This project implements a basic linear regression model to predict the **price of a car based on its mileage**. It's designed as a simple educational tool to understand how linear regression works under the hood, without relying on high-level machine learning libraries.

## 📌 Project Overview

You will build **two Python programs**:

1. **Predictor** – Given a mileage input, it estimates the price using learned parameters (`theta0`, `theta1`).
2. **Trainer** – Reads a dataset (mileage vs. price) and learns the optimal values of `theta0` and `theta1` using **gradient descent**.

---
## 🏁 How to Run

### 1. Prepare Your Dataset

Create or replace a CSV file named `data.csv` with the following structure:
| mileage | price |
|---------|-------|
| 120000  | 5000  |
| 100000  | 5500  |
| ...     | ...   |

### 2. Train the model
Run the training program:
```./train.py```
To generate a ```thetas.json``` file that contains the weights.

### 3. Predict
Run:
```./predict.py``` and input a value for ```mileage``` in order to get a value for ```price```
