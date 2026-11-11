import heapq
import math
import numpy as np
from csp.sudoku import solve, PEERS


def make_actions(grid, probs, k=12):
    """For the k least-confident cells: {cell: {new_digit: cost}}."""
    cells = [(r, c) for r in range(9) for c in range(9) if grid[r][c]]
    cells.sort(key=lambda rc: probs[rc[0], rc[1], grid[rc[0]][rc[1]] - 1])
    actions = {}
    for r, c in cells[:k]:
        p_old = max(probs[r, c, grid[r][c] - 1], 1e-12)
        top3 = np.argsort(-probs[r, c])[:3] + 1
        alts = {}
        for d in top3:
            d = int(d)
            if d != grid[r][c]:
                p_new = max(probs[r, c, d - 1], 1e-12)
                alts[d] = max(0.0, math.log(p_old) - math.log(p_new))
        actions[(r, c)] = alts
    return actions


def find_conflicts(grid):
    """Pairs of cells that see each other and hold the same digit."""
    out = []
    for a in PEERS:
        for b in PEERS[a]:
            if a < b and grid[a[0]][a[1]] and grid[a[0]][a[1]] == grid[b[0]][b[1]]:
                out.append((a, b))
    return out


def correct(grid, probs, method="ucs", max_corrections=3, k=12):
    """method: 'ucs', 'greedy' or 'astar'."""
    stats = {"generated": 1, "expanded": 0, "solver_calls": 0}
    actions = make_actions(grid, probs, k)

    def apply(state):
        g = [row[:] for row in grid]
        for r, c, d in state:
            g[r][c] = d
        return g

    def heuristic(g, state):
        # Every duplicate pair needs at least one of its cells changed.
        # Lower bound = the worst pair's cheapest possible fix.
        if method == "ucs":
            return 0.0
        changed = {(r, c) for r, c, _ in state}
        worst = 0.0
        for a, b in find_conflicts(g):
            options = [cost for cell in (a, b) if cell not in changed
                       for cost in actions.get(cell, {}).values()]
            if not options:
                return math.inf
            worst = max(worst, min(options))
        return worst

    def priority(cost, h):
        if method == "ucs":
            return (cost, cost)
        if method == "greedy":
            return (h, cost)
        return (cost + h, cost)

    heap = [((0.0, 0.0), 0.0, ())]
    seen = {frozenset()}
    while heap:
        _, cost, state = heapq.heappop(heap)
        stats["expanded"] += 1
        g = apply(state)
        stats["solver_calls"] += 1
        sol, _ = solve(g, use_fc=True, use_ac3=True)
        if sol is not None:
            log = [((r, c), grid[r][c], d, actions[(r, c)][d]) for r, c, d in state]
            return {"found": True, "log": log, "cost": cost, "grid": g, "stats": stats}
        if len(state) == max_corrections:
            continue
        changed = {(r, c) for r, c, _ in state}
        for (r, c), alts in actions.items():
            if (r, c) in changed:
                continue
            for d, step in alts.items():
                new_state = tuple(sorted(state + ((r, c, d),)))
                key = frozenset(new_state)
                if key in seen:
                    continue
                seen.add(key)
                new_cost = cost + step
                h = heuristic(apply(new_state), new_state)
                if h == math.inf:
                    continue
                heapq.heappush(heap, (priority(new_cost, h), new_cost, new_state))
                stats["generated"] += 1
    return {"found": False, "log": [], "cost": None, "grid": None, "stats": stats}
