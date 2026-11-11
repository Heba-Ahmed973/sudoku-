from itertools import product
from csp.sudoku import ac3, solve
from search.correct import correct, find_conflicts


def ac3_conflict(grid):
    """True if AC-3 alone proves the grid is broken (a domain becomes empty)."""
    domains = {(r, c): ({grid[r][c]} if grid[r][c] else set(range(1, 10)))
               for r, c in product(range(9), range(9))}
    return not ac3(domains)


def least_confident(grid, probs, n=5):
    cells = [(r, c) for r in range(9) for c in range(9) if grid[r][c]]
    cells.sort(key=lambda rc: probs[rc[0], rc[1]].max())
    return [(rc, round(float(probs[rc[0], rc[1]].max()), 2)) for rc in cells[:n]]


def to_grid(sol):
    return [[sol[(r, c)] for c in range(9)] for r in range(9)]


def run_pipeline(grid, probs, mode="nn", method="ucs"):
    """mode 'nn' = use NN probabilities to correct, 'off' = no correction."""
    out = {
        "ac3_conflict": ac3_conflict(grid),       # step 1: AC-3 looks first
        "conflicts": find_conflicts(grid),
        "log": [], "cost": None, "stats": None,
        "solution": None, "corrected_grid": None, "least_confident": [],
    }
    sol, _ = solve(grid, use_fc=True, use_ac3=True)   # step 2: CSP tries as read
    if sol is not None:
        out.update(status="solved", solution=to_grid(sol), corrected_grid=grid)
        return out
    if mode == "off":
        out["status"] = "no solution (NN off: no correction tried)"
        return out
    res = correct(grid, probs, method)                # step 3: Search fixes it
    out["stats"] = res["stats"]
    if res["found"]:                                  # step 4: CSP solves fixed grid
        sol, _ = solve(res["grid"], use_fc=True, use_ac3=True)
        out.update(status="solved", log=res["log"], cost=res["cost"],
                   solution=to_grid(sol), corrected_grid=res["grid"])
    else:
        out["status"] = "unreadable"
        out["least_confident"] = least_confident(grid, probs)
    return out
