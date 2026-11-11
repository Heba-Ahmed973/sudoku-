import math
import numpy as np
from csp.generator import make_puzzle
from search.correct import correct

puzzle, _ = make_puzzle(seed=1, givens=30)

# find a row with at least 2 given digits
for r in range(9):
    given = [c for c in range(9) if puzzle[r][c]]
    if len(given) >= 2:
        c, other = given[0], given[1]
        break
true_digit = puzzle[r][c]
wrong = puzzle[r][other]          # a digit already in this row -> duplicate

grid = [row[:] for row in puzzle]
grid[r][c] = wrong

probs = np.full((9, 9, 9), 0.001)
for rr in range(9):
    for cc in range(9):
        if grid[rr][cc]:
            probs[rr, cc, grid[rr][cc] - 1] = 0.98
probs[r, c, wrong - 1] = 0.52
probs[r, c, true_digit - 1] = 0.45

expected = math.log(0.52 / 0.45)
for method in ["ucs", "greedy", "astar"]:
    res = correct(grid, probs, method)
    cell, old, new, cost = res["log"][0]
    ok = (cell == (r, c) and new == true_digit and abs(cost - expected) < 1e-9)
    print(method, "PASS" if ok else "FAIL", res["log"], res["stats"])
