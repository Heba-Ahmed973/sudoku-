from itertools import product

def make_peers():
    P = {}
    for r, c in product(range(9), range(9)):
        s = {(r, i) for i in range(9)} | {(i, c) for i in range(9)}
        br, bc = 3 * (r // 3), 3 * (c // 3)
        s |= {(br + i, bc + j) for i in range(3) for j in range(3)}
        s.discard((r, c))
        P[(r, c)] = s
    return P

PEERS = make_peers()

def ac3(domains):
    queue = [(a, b) for a in domains for b in PEERS[a]]
    while queue:
        a, b = queue.pop()
        if len(domains[b]) == 1:
            v = next(iter(domains[b]))
            if v in domains[a]:
                domains[a].discard(v)
                if not domains[a]:
                    return False          # conflict found
                queue += [(x, a) for x in PEERS[a] if x != b]
    return True

def solve(grid, use_fc=False, use_ac3=False):
    """grid: 9x9 list, 0 = empty. Returns (solution or None, stats)."""
    stats = {"assignments": 0, "backtracks": 0}
    domains = {(r, c): ({grid[r][c]} if grid[r][c] else set(range(1, 10)))
               for r, c in product(range(9), range(9))}
    assignment = {c: grid[c[0]][c[1]] for c in domains if grid[c[0]][c[1]]}

    for cell, v in assignment.items():            # givens must not clash
        if any(assignment.get(p) == v for p in PEERS[cell]):
            return None, stats
    if use_ac3 and not ac3(domains):
        return None, stats
    if use_fc:
        for cell, v in assignment.items():
            for p in PEERS[cell]:
                if p not in assignment:
                    domains[p].discard(v)
                    if not domains[p]:
                        return None, stats

    def search(assign, doms):
        if len(assign) == 81:
            return assign
        cell = min((c for c in doms if c not in assign), key=lambda c: len(doms[c]))
        for v in sorted(doms[cell]):
            if any(assign.get(p) == v for p in PEERS[cell]):
                continue
            stats["assignments"] += 1
            new_assign, new_doms = {**assign, cell: v}, doms
            if use_fc:
                new_doms = {c: set(d) for c, d in doms.items()}
                new_doms[cell] = {v}
                if any(not new_doms[p].discard(v) and not new_doms[p]
                       for p in PEERS[cell] if p not in new_assign):
                    stats["backtracks"] += 1
                    continue
            result = search(new_assign, new_doms)
            if result:
                return result
            stats["backtracks"] += 1
        return None

    sol = search(assignment, domains)
    return sol, stats