#!/home/ljurdant/.venv/ft_linear_regression/bin/python3


import json
import numpy as np
from LinearRegression import LinearRegression

if __name__ == "__main__":

    print("Predict the price of a car based on its mileage.")
    print("Please enter a mileage in km:")

    valid = False
    while not valid:
        try:
            mileage = float(input())
            valid = True
        except ValueError:
            print("Invalid input. Please enter a number:")

    thetas = np.array([0, 0])
    try:
        with open("thetas.json") as f:
            thetas = json.load(f)
            thetas = np.array(thetas)

    except FileNotFoundError:
        print("No thetas.json file found. Thetas initialized to 0.")
    except json.JSONDecodeError:
        print("Error while reading thetas.json file. Thetas initialized to 0.")

    lr = LinearRegression(thetas=thetas)
    price = lr.predict(np.array([mileage]))
    print("Estimated price:", price)
