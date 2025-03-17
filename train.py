#!/home/ljurdant/.venv/ft_linear_regression/bin/python3

import pandas as pd
import numpy as np
import json


def minmax(x):
    if isinstance(x, np.ndarray):
        return (x - min(x)) / (max(x) - min(x))


def gradient(x, y, thetas):
    theta0 = 1 / x.shape[0] * np.sum(thetas[0] + thetas[1] * x - y)
    theta1 = 1 / x.shape[0] * np.sum((thetas[0] + thetas[1] * x - y) * x)
    return np.array([[theta0], [theta1]])


def train(x, y, thetas):
    max_iter = 1000000

    alpha = 0.01

    for i in range(max_iter):
        gradient_values = gradient(x, y, thetas)
        thetas = thetas - alpha * gradient_values

    return thetas


data = pd.read_csv("datasets/data.csv")
x = minmax(np.array(data["km"]))
y = minmax(np.array(data["price"]))
thetas = np.random.rand(2).reshape(2, 1)

thetas = train(x, y, thetas)

json.dump(thetas.tolist(), open("thetas.json", "w"))
