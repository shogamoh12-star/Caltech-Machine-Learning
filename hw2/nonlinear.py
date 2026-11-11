import matplotlib.pyplot as plt
import numpy as np


def transform_one(matrix):
    transformed_matrix = np.column_stack((
        np.ones(matrix.shape[0]),
        matrix[:, 1] ** 2,
        matrix[:, 2] ** 2
    ))
    return transformed_matrix

def transform_two(matrix):
    transformed_matrix = np.column_stack((
        np.ones(matrix.shape[0]),
        matrix[:, 1],
        matrix[:, 2],
        matrix[:, 1] * matrix[:, 2],
        matrix[:, 1] ** 2,
        matrix[:, 2] ** 2
    ))
    return transformed_matrix

def check_e_in(y_vector, hypothesis, x_matrix, N):
    disagreement = 0
    for i in range(0, N):
        if (y_vector[i] != np.sign(np.dot(x_matrix[i], hypothesis))):
            disagreement += 1

    return disagreement / N

N = 1000
def generate(target):
    x_matrix = np.column_stack((
            np.ones(N),
            np.random.uniform(-1, 1, (N, 2))
        ))
    transformed_matrix = transform_one(x_matrix)
    y_vector = np.sign(transformed_matrix @ target)
    flip = np.random.random(N) < 0.1
    y_vector[flip] = y_vector[flip] * -1
    return x_matrix, y_vector

def check_e_out(hypothesis):
    disagreement = 0
    x_matrix, y_vector = generate([-0.6, 1, 1])
    x_matrix = transform_two(x_matrix)
    for i in range(0, 1000):
        if (y_vector[i] != np.sign(np.dot(x_matrix[i], hypothesis))):
             disagreement += 1

    return disagreement / 1000


e_ins = np.zeros(1000)
for i in range(0, 1000):
    x_matrix, y_vector = generate([-0.6, 1, 1])
    pseudo_inverse_x = np.linalg.inv(x_matrix.T @ x_matrix) @ (x_matrix.T)
    hypothesis = pseudo_inverse_x @ y_vector
    e_ins[i] = check_e_in(y_vector, hypothesis, x_matrix, N)

print("Avg E in (normal lin regression): " + str(np.average(e_ins)))

runs = 1000
hypothesises = np.zeros((runs, 6))
e_outs = np.zeros(runs)
for i in range(0, runs):
    x_matrix, y_vector = generate([-0.6, 1, 1])
    new_matrix = transform_two(x_matrix)
    pseudo_inverse_x = np.linalg.inv(new_matrix.T @ new_matrix) @ (new_matrix.T)
    hypothesises[i] = pseudo_inverse_x @ y_vector
    e_outs[i] = check_e_out(hypothesises[i])


print("Hypothesis after transforming points: ")
for i in range(0, 6):
    print(str(np.average(hypothesises[:, i])))
print("Avg E out (using lin regression after applying transform): " + str(np.average(e_outs)))