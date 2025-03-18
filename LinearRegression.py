import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, thetas=np.zeros(2), max_iter=1300, alpha=1e-2):
        self.max_iter = max_iter
        self.alpha = alpha
        self.thetas = thetas
        self.x_normalizer = None

    def predict(self, x, normalized=False):
        if not normalized:
            thetas = self.denormalize_thetas()
        else:
            thetas = self.thetas

        return thetas[0] + thetas[1] * x

    def gradient(self, x, y):
        theta0 = 1 / x.shape[0] * np.sum(self.predict(x, normalized=True) - y)
        theta1 = 1 / x.shape[0] * np.sum((self.predict(x, normalized=True) - y) * x)
        return np.array([theta0, theta1])

    def train(self, x, y):

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(x, y)

        y_pred = self.predict(x)
        (line1,) = ax.plot(x, y_pred, color="red")

        plt.xlabel("km")
        plt.ylabel("price")
        plt.show()

        self.x_normalizer = self.Normalizer()
        x_normalized = self.x_normalizer.normalize(x)

        for i in range(self.max_iter + 1):
            gradient_values = self.gradient(x_normalized, y)
            self.thetas -= self.alpha * gradient_values
            if i % 10 == 0:
                y_pred = self.predict(x)
                line1.set_ydata(y_pred)
                plt.title(f"Iteration {i}  Loss: {self.loss(x, y):.2f}")
                fig.canvas.draw()
                fig.canvas.flush_events()
        plt.pause(1)
        return self.denormalize_thetas()

    def denormalize_thetas(self):
        if self.x_normalizer is None:
            return self.thetas

        theta0_denorm = (
            self.thetas[0]
            - self.thetas[1] * self.x_normalizer.mean / self.x_normalizer.std
        )
        theta1_denorm = self.thetas[1] / self.x_normalizer.std

        return np.array([theta0_denorm, theta1_denorm])

    def loss(self, x, y):
        return 1 / (x.shape[0]) * np.sum((y - self.predict(x)) ** 2)

    class Normalizer:
        def __init__(self):
            self.mean = 0
            self.std = 0

        def normalize(self, x):
            self.mean = np.mean(x)
            self.std = np.std(x)
            return (x - self.mean) / self.std
