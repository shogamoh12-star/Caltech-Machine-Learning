# Caltech-Machine-Learning

Problems and solutions to the Caltech Machine Learning CS 156 course
(*Learning From Data*, Yaser Abu-Mostafa).

## Structure

Each homework is in its own folder with its own Python virtual environment:

| Folder | Topic |
|--------|-------|
| `hw1/` | Perceptron Learning Algorithm |
| `hw2/` | Hoeffding Inequality |

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
