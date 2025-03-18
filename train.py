#!/home/ljurdant/.venv/ft_linear_regression/bin/python3

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

data = pd.read_csv("datasets/data.csv")
x = np.array(data["km"])
y = np.array(data["price"])

lr = LinearRegression()
thetas = lr.train(x, y)

print("Thetas:", thetas)

json.dump(thetas.tolist(), open("thetas.json", "w"))
