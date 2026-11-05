# Caltech-Machine-Learning

Problems and solutions to the Caltech Machine Learning CS 156 course
(*Learning From Data*, Yaser Abu-Mostafa).

## Structure

Each homework is in its own folder with its own Python virtual environment:

| Folder | Topic |
|--------|-------|
| `hw1/` | Perceptron Learning Algorithm |

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

## Running a homework

```bash
cd hw1
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib ipykernel
python perceptron.py
```
