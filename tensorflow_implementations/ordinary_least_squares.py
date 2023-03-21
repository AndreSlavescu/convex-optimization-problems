import tensorflow as tf
import cvxpy as cp
from cvxpylayers.tensorflow import CvxpyLayer

"""
Ordinary Least Squares
"""

class OLS(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = tf.Variable(tf.random.normal((input_size, output_size)))
        self.b = tf.Variable(tf.random.normal((output_size,)))
        self.cvxpy_layer = self.create_cvxpy_layer(input_size, output_size)

    def call(self, inputs):
        X, y = inputs
        solution, = self.cvxpy_layer([X, y])
        return solution

    def create_cvxpy_layer(self, input_size, output_size):
        X = cp.Variable((None, input_size))
        y = cp.Variable((None, output_size))
        constraints = [self.W >= 0]
        objective = cp.Minimize(cp.sum_squares(X @ self.W + self.b - y))
        problem = cp.Problem(objective, constraints)

        assert problem.is_dpp()

        cvxpy_layer = CvxpyLayer(problem, [X, y], [self.W, self.b])
        return cvxpy_layer

# Test
X = tf.random.normal((10, 5))
y = tf.random.normal((10, 1))
model = OLS(input_size=5, output_size=1)
solution = model([X, y])
print(solution)
