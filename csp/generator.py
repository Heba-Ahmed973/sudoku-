import random
from itertools import product
from csp.sudoku import PEERS


def count_solutions(grid, limit=2):
    """Count solutions, but stop once we find `limit` of them."""
    grid = [row[:] for row in grid]
    count = 0

    def go():
        nonlocal count
        best = None
        for r, c in product(range(9), range(9)):
            if grid[r][c] == 0:
                used = {grid[pr][pc] for pr, pc in PEERS[(r, c)]}
                cand = [v for v in range(1, 10) if v not in used]
                if not cand:
                    return              # dead end
                if best is None or len(cand) < len(best[2]):
                    best = (r, c, cand)
        if best is None:                # no empty cells = one solution found
            count += 1
            return
        r, c, cand = best
        for v in cand:
            grid[r][c] = v
            go()
            grid[r][c] = 0
            if count >= limit:
                return

    go()
    return count


def full_grid(rng):
    """Build a complete valid Sudoku using random digit order."""
    grid = [[0] * 9 for _ in range(9)]

    def fill():
        for r, c in product(range(9), range(9)):
            if grid[r][c] == 0:
                used = {grid[pr][pc] for pr, pc in PEERS[(r, c)]}
                options = [v for v in range(1, 10) if v not in used]
                rng.shuffle(options)
                for v in options:
                    grid[r][c] = v
                    if fill():
                        return True
                grid[r][c] = 0
                return False
        return True

    fill()
    return grid


def make_puzzle(seed, givens=30):
    """Return (puzzle, solution). Puzzle has exactly one solution."""
    rng = random.Random(seed)           # same seed = same puzzle
    solution = full_grid(rng)
    puzzle = [row[:] for row in solution]

    cells = list(product(range(9), range(9)))
    rng.shuffle(cells)

    filled = 81
    for r, c in cells:
        if filled <= givens:
            break
        old = puzzle[r][c]
        puzzle[r][c] = 0
        if count_solutions(puzzle) != 1:
            puzzle[r][c] = old          # not unique, put it back
        else:
            filled -= 1
    return puzzle, solution