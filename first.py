"""Simple machine learning example: linear regression from scratch."""


def train_linear_regression(data, learning_rate=0.01, epochs=1000):
    slope = 0.0
    intercept = 0.0
    total_points = len(data)

    for _ in range(epochs):
        slope_gradient = 0.0
        intercept_gradient = 0.0

        for x_value, y_value in data:
            prediction = (slope * x_value) + intercept
            error = y_value - prediction

            slope_gradient += (-2 / total_points) * x_value * error
            intercept_gradient += (-2 / total_points) * error

        slope -= learning_rate * slope_gradient
        intercept -= learning_rate * intercept_gradient

    return slope, intercept


def predict(x_value, slope, intercept):
    return (slope * x_value) + intercept


def main():
    training_data = [
        (1, 52),
        (2, 57),
        (3, 63),
        (4, 69),
        (5, 74),
        (6, 80),
    ]

