# Caltech-Machine-Learning

Problems and solutions to the Caltech Machine Learning CS 156 course
(*Learning From Data*, Yaser Abu-Mostafa).

## Structure

Each homework is in its own folder with its own Python virtual environment:

| Folder | Topic |
|--------|-------|
| `hw1/` | Perceptron Learning Algorithm |
| `hw2/` | Hoeffding Inequality, Linear Regression, Nonlinear Transformation |

## Homeworks

### HW1: Perceptron Learning Algorithm

`hw1/perceptron.py` simulates the Perceptron Learning Algorithm (PLA) on
randomly generated, linearly separable data in [-1, 1] × [-1, 1]:

- Picks a random target line and labels N random points by which side of it they fall on
- Runs PLA from all-zero weights, updating on a randomly chosen misclassified point until every point is classified correctly
- Estimates how often the final hypothesis disagrees with the target, using 1,000 fresh test points
- Repeats this 1,000 times (N = 10) and prints the average number of iterations and the average disagreement

To watch the hypothesis line converge, call `run(10, True)` once instead of
running the 1,000-run loop. With plotting on, every run opens its own window.

### HW2: Hoeffding Inequality

`hw2/hoeffding.py` simulates flipping 1,000 fair coins 10 times each:

- In each experiment, records the fraction of heads for three coins: the first coin (ν₁), a randomly chosen coin (ν_rand), and the coin with the fewest heads (ν_min, taking the earlier coin in a tie)
- Repeats the experiment 100,000 times, using numpy to generate each experiment's flips in one array
- Plots the three distributions as side-by-side histograms, one bar per possible value (0, 0.1, …, 1.0), with the fraction of runs on the y-axis

The 100,000 runs take a little while; the plot window opens when they finish.
After you close it, the script prints the average of ν_min.

#### Results

**Average $\nu_{\min}$:** the simulation gives $\approx 0.038$, closest to **0.01**.
This matches an exact estimate: no coin lands all tails with probability
$(1 - 1/1024)^{1000} \approx 0.38$, and in those runs $\nu_{\min}$ is almost always $0.1$,
so $\mathbb{E}[\nu_{\min}] \approx 0.1 \times 0.38 \approx 0.038$.

**Which coins satisfy the single-bin Hoeffding bound**
$\mathbb{P}(|\nu - \mu| > \varepsilon) \le 2e^{-2\varepsilon^2 N}$? **$c_1$ and $c_{\text{rand}}$**.
Both are fixed before the flips are seen, so each is a single bin. $c_{\min}$ is picked *after*
seeing the outcomes, which pushes $\nu$ towards 0 and away from $\mu = 0.5$ far more often than
the bound allows. That is the multiple-bin case, which needs a union-bound factor of $M = 1000$.

### HW2: Linear Regression

`hw2/lin_regression.py` uses Linear Regression for classification on the same
kind of data as HW1 (a random target line in [-1, 1] × [-1, 1]):

- `regression(N, plot)` labels N random points with the target, finds the weights with the pseudo-inverse, and returns E_in (fraction of training points misclassified) and E_out (estimated on 1,000 fresh points). With `plot=True` it draws the points, the target line and the hypothesis line
- `pla(...)` runs the Perceptron Learning Algorithm starting from the regression weights instead of zeros, and counts the iterations until every point is classified correctly
- The script repeats this 1,000 times with N = 10 and prints the average E_in, E_out and number of PLA iterations. Change the `regression(10, False)` call to `regression(100, False)` for N = 100

#### Results

Averages over 1,000 runs with N = 10:

| E_in | E_out | PLA iterations |
|------|-------|----------------|
| ≈ 0.03 | ≈ 0.1 | ≈ 5 |

Starting PLA from the Linear Regression weights roughly halves the number of
iterations it needs to converge, compared with HW1:

| PLA starting weights (N = 10, averaged over 1,000 runs) | Average iterations |
|---|---|
| All zeros (HW1, `hw1/perceptron.py`) | ≈ 9 |
| Linear Regression weights (HW2) | ≈ 5 |

The regression weights already separate most of the points, so PLA only has to
fix the few that are left.

### HW2: Nonlinear Transformation

`hw2/nonlinear.py` uses Linear Regression on a circular target,
$f(x_1, x_2) = \text{sign}(x_1^2 + x_2^2 - 0.6)$, with N = 1,000 points in
[-1, 1] × [-1, 1] and about 10% of the labels flipped as noise:

- Fits Linear Regression on the raw features (1, x₁, x₂) and averages E_in over 1,000 runs
- Transforms the features to (1, x₁, x₂, x₁x₂, x₁², x₂²), fits Linear Regression again, and averages the weights over 1,000 runs
- Estimates E_out of each transformed hypothesis on 1,000 fresh noisy points and averages it

#### Results

From one run of the script:

| | Output |
|---|---|
| Average E_in, raw features | ≈ 0.50 |
| Average weights, transformed (w₀ … w₅) | ≈ (−0.99, 0.00, 0.00, 0.00, 1.56, 1.56) |
| Average E_out, transformed | ≈ 0.13 |

- **Without a transform** the classifier is no better than guessing (E_in ≈ 0.5, closest to **0.5**). A straight line can't separate the inside of a circle from the outside.
- **With the transform** the hypothesis is roughly $\text{sign}(-1 + 1.5x_1^2 + 1.5x_2^2)$: the linear and cross terms average out to about 0. That is a circle of radius² ≈ 0.64, close to the target's 0.6.
- **E_out with the transform** is ≈ 0.13, closest to **0.1**. Most of that error comes from the 10% label noise, which no hypothesis can fit.

## Running a homework

```bash
cd hw1
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib ipykernel
python perceptron.py
```

Replace `hw1` and `perceptron.py` with another homework's folder and script
(e.g. `hw2` and `hoeffding.py`) to run that one.
