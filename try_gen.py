from csp.generator import make_puzzle
from csp.sudoku import solve

puzzle, solution = make_puzzle(seed=1, givens=30)

for row in puzzle:
    print(" ".join(str(x) if x else "." for x in row))

sol, stats = solve(puzzle, use_fc=True, use_ac3=True)
ok = sol is not None and all(sol[(r, c)] == solution[r][c]
                             for r in range(9) for c in range(9))
print("Solver found the right answer:", ok)
print(stats)
