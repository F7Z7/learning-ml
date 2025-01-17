import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        # Calculate slope (beta1)
        self.slope = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)

        # Calculate intercept (beta0)
        self.intercept = mean_y - self.slope * mean_x

        print(f"Slope: {self.slope} and Intercept: {self.intercept}")

    def predict(self, x):
        return self.slope * x + self.intercept


# Example dataset
x = [1, 2, 3, 4, 5]
y = [2.1, 4.2, 6.1, 8.0, 10.1]

# Create an instance of the SimpleLinearRegression class
regression = LinearRegression()
regression.fit(x, y)

# Predicting new values
new_x = 6
predictions = regression.predict(new_x)
print(f"Predicted value for x = {new_x}: {predictions}")
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y, color='red', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
