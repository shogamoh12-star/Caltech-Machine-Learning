import random
import numpy as np
import matplotlib.pyplot as plt

def run():
    flips = np.random.randint(0, 2, size = (1000, 10))
    nu_array = flips.mean(axis = 1)
    return nu_array[0], nu_array.min(), nu_array[random.randint(0, 999)]

size = 100000
nu_one = np.zeros(size)
nu_min = np.zeros(size)
nu_rand = np.zeros(size)
for i in range(0, size):
    nu_one[i], nu_min[i], nu_rand[i] = run()

fig, axes = plt.subplots(1, 3, sharex = True, sharey = True)
bins = np.linspace(-0.05, 1.05, 12)
axes[0].hist(nu_one, bins, color = "blue", weights=np.ones(size) / size)
axes[1].hist(nu_min, bins, color = "blue", weights=np.ones(size) / size)
axes[2].hist(nu_rand, bins, color = "blue", weights=np.ones(size) / size)
axes[0].set_title("Nu One")
axes[1].set_title("Nu Min")
axes[2].set_title("Nu Random")
axes[0].set_ylabel("Fraction of runs")
plt.show()

print(np.mean(nu_min))
