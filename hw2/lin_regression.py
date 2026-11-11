import random
import numpy as np
import matplotlib.pyplot as plt


def regression(N, plot):
    x_matrix = np.column_stack((
        np.ones(N),
        np.random.uniform(-1, 1, (N, 2))
    ))

    point_1 = [random.uniform(-1, 1), random.uniform(-1, 1)]
    point_2 = [random.uniform(-1, 1), random.uniform(-1, 1)]

    m = (point_1[1] - point_2[1]) / (point_1[0] - point_2[0])
    target_function = np.array([-1 * point_1[1] + m * point_1[0], -1 * m, 1])
    above = random.choice([True, False])

    y_vector = np.zeros(N)
    for i in range(0, N):
        y_vector[i] = np.sign(np.dot(x_matrix[i], target_function))
        if not above:
            y_vector[i] = y_vector[i] * -1

    pseudo_inverse_x = np.linalg.inv(x_matrix.T @ x_matrix) @ (x_matrix.T)

    hypothesis = pseudo_inverse_x @ y_vector
    if plot:
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axline((point_1[0], point_1[1]), (point_2[0], point_2[1]), label = "target")
        for i in range(0, N):
            if y_vector[i] > 0:
                plt.scatter(x_matrix[i][1], x_matrix[i][2], color = "blue")
            else:
                plt.scatter(x_matrix[i][1], x_matrix[i][2], color = "red")

        plt.axline((0, -hypothesis[0] / hypothesis[2]), (-hypothesis[0] / hypothesis[1], 0), color = "orange", label = "hypothesis")

        plt.legend()
        plt.show()
    
    return check_e_out(target_function, hypothesis, above), check_e_in(y_vector, hypothesis, x_matrix, N), x_matrix, hypothesis, y_vector


def pla(x_matrix, y_vector, hypothesis, N):
    wrong = []
    iterations = 0
    for i in range(0, N):
            if (y_vector[i] != np.sign(np.dot(x_matrix[i], hypothesis))):
                wrong.append(i)
    
    while (wrong != []):
            fix = random.choice(wrong)
            hypothesis = hypothesis + np.dot(y_vector[fix], x_matrix[fix])
            iterations += 1
            wrong = []
            for i in range(0, N):
                if (y_vector[i] != np.sign(np.dot(x_matrix[i], hypothesis))):
                    wrong.append(i)

    return iterations


def check_e_out(target, hypothesis, above):
    disagreement = 0
    test_x = np.column_stack((
        np.ones(1000),
        np.random.uniform(-1, 1, (1000, 2))
    ))

    for i in range(0, 1000):
        if (above and np.sign(np.dot(test_x[i], target)) != np.sign(np.dot(test_x[i], hypothesis))):
             disagreement += 1
        if (not above and np.sign(np.dot(test_x[i], target)) == np.sign(np.dot(test_x[i], hypothesis))):
            disagreement += 1

    return disagreement / 1000

def check_e_in(y_vector, hypothesis, x_matrix, N):
    disagreement = 0
    for i in range(0, N):
        if (y_vector[i] != np.sign(np.dot(x_matrix[i], hypothesis))):
            disagreement += 1

    return disagreement / N



e_out_array = np.zeros(1000)
e_in_array = np.zeros(1000)
iterations_array = np.zeros(1000)
for i in range(0, 1000):
    e_out, e_in, x_matrix, hypothesis, y_vector = regression(10, False)
    e_out_array[i] = e_out
    e_in_array[i] = e_in
    iterations_array[i] = pla(x_matrix, y_vector, hypothesis, 10)
    

print("E_in: " + str(np.average(e_in_array)))
print("E_out: " + str(np.average(e_out_array)))
print("Iterations: " + str(np.average(iterations_array)))

