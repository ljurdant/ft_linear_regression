import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, max_iter=10000, alpha=1e-3):
        self.max_iter = max_iter
        self.alpha = alpha
        self.thetas = np.zeros(2)
        self.x_normalizer = None
        self.y_normalizer = None

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

        plt.show()

        self.x_normalizer = self.Normalizer()
        self.y_normalizer = self.Normalizer()
        x_normalized = self.x_normalizer.normalize(x)
        y_normalized = self.y_normalizer.normalize(y)

        for i in range(self.max_iter):
            gradient_values = self.gradient(x_normalized, y_normalized)
            self.thetas -= self.alpha * gradient_values

            if i % 100 == 0:
                y_pred = self.predict(x)
                line1.set_ydata(y_pred)
                fig.canvas.draw()
                # print(self.thetas, self.denormalize_thetas())
                fig.canvas.flush_events()
        # while 1:
        plt.pause(1)
        return self.denormalize_thetas()

    def denormalize_thetas(self):
        if self.x_normalizer is None or self.y_normalizer is None:
            return self.thetas

        theta1_denorm = self.thetas[1] * (self.y_normalizer.std / self.x_normalizer.std)
        theta0_denorm = self.y_normalizer.mean - theta1_denorm * self.x_normalizer.mean

        return np.array([theta0_denorm, theta1_denorm])

    class Normalizer:
        def __init__(self):
            self.mean = 0
            self.std = 0

        def normalize(self, x):
            self.mean = np.mean(x)
            self.std = np.std(x)
            return (x - self.mean) / self.std
