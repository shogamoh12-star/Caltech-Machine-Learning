# Perceptron Learning Algorithm - HW1
import random
import numpy as np
import matplotlib.pyplot as plt


def run(sample_size, plot):

    class Point():
        def __init__(self):
            self.x = random.uniform(-1, 1)
            self.y = random.uniform(-1, 1)
            if (self.x * target_function_slope + target_function_intercept >= self.y):
                self.output = target_function_above_or_below
            else:
                self.output = target_function_above_or_below * -1

    def check_disagreement(hypothesis):
        points = []
        total_disagreement = 0
        number_of_points = 1000
        for i in range(0, number_of_points):
            points.append(Point())
            if(points[i].output != hypothesis.output(points[i])):
                total_disagreement += 1
        return total_disagreement / number_of_points

    class Hypothesis():
        def __init__(self):
            self.w1 = 0
            self.w2 = 0
            self.threshold = 0
        def output(self, Point):
            return np.sign(self.w1 * Point.x + self.w2 * Point.y + self.threshold)


    point1_x = random.uniform(-1, 1)
    point1_y = random.uniform(-1, 1)
    point2_x = random.uniform(-1, 1)
    point2_y = random.uniform(-1, 1)
    target_function_slope = (point1_y - point2_y) / (point1_x - point2_x)
    target_function_intercept = point1_y - target_function_slope * point1_x
    target_function_above_or_below = random.choice([-1, 1])
    hypothesis_function = Hypothesis()
    sample = []
    xs = []
    ys = []
    colours = []
    for i in range(0, sample_size):
        point = Point()
        sample.append(point)
        xs.append(sample[i].x)
        ys.append(sample[i].y)
        if (sample[i].output == 1):
            colours.append("blue")
        if (sample[i].output == -1):
            colours.append("red")
            
    if (plot == True):
        plt.axline((point1_x, point1_y), slope = target_function_slope, color = "green", label = "Target")
        plt.scatter(xs, ys, c=colours)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
    


    wrong = []

    for i in range(0, sample_size):
        if (sample[i].output != hypothesis_function.output(sample[i])):
            wrong.append(sample[i])


    iterations = 0

    line = None

    while (wrong != []):
            fix = random.choice(wrong)
            hypothesis_function.w1 = hypothesis_function.w1 + fix.output * fix.x
            hypothesis_function.w2 = hypothesis_function.w2 + fix.output * fix.y
            hypothesis_function.threshold = hypothesis_function.threshold + fix.output
            iterations += 1
            if (plot == True):
                line = plt.axline((0, -1 * hypothesis_function.threshold / hypothesis_function.w2), slope =  -1 * hypothesis_function.w1 / hypothesis_function.w2, color = "purple")
                plt.pause(0.1)
                line.remove()
            wrong = []
            for point in sample:
                if point.output != hypothesis_function.output(point):
                    wrong.append(point)

    if (plot == True):
        plt.axline((0, -1 * hypothesis_function.threshold / hypothesis_function.w2), slope =  -1 * hypothesis_function.w1 / hypothesis_function.w2, color = "purple", label = "Hypothesis")
        plt.legend()
        plt.show()
    return iterations, check_disagreement(hypothesis_function)

avg_iteration = 0
avg_disagreement = 0
for i in range(0, 1000):
    output = run(10, False)
    avg_iteration += output[0]
    avg_disagreement += output[1]

print("Average iterations: ", avg_iteration / 1000)
print("Average disagreement: ", avg_disagreement / 1000)


