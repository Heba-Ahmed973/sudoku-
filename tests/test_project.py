import math
import numpy as np

from csp.sudoku import solve
from csp.generator import make_puzzle, count_solutions
from search.correct import correct
from integration.pipeline import run_pipeline, ac3_conflict

PUZZLE = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
ANSWER = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


def to_grid(s):
    return [[int(s[r * 9 + c]) for c in range(9)] for r in range(9)]


def flat(sol):
    return "".join(str(sol[(r, c)]) for r in range(9) for c in range(9))


def flat_grid(g):
    return "".join(str(x) for row in g for x in row)


# ---------------- 5 NORMAL ----------------
def test_normal_1_backtracking_solves_known_puzzle():
    sol, stats = solve(to_grid(PUZZLE))
    assert flat(sol) == ANSWER


def test_normal_2_fc_and_ac3_give_same_answer():
    for fc, ac in [(True, False), (True, True)]:
        sol, _ = solve(to_grid(PUZZLE), use_fc=fc, use_ac3=ac)
        assert flat(sol) == ANSWER


def test_normal_3_fc_and_ac3_do_less_work():
    _, plain = solve(to_grid(PUZZLE))
    _, smart = solve(to_grid(PUZZLE), use_fc=True, use_ac3=True)
    assert smart["assignments"] <= plain["assignments"]


def test_normal_4_generator_unique_and_seeded():
    p1, s1 = make_puzzle(3, 30)
    p2, _ = make_puzzle(3, 30)
    p3, _ = make_puzzle(4, 30)
    assert p1 == p2                      # same seed = same puzzle
    assert p1 != p3                      # different seed = different puzzle
    assert count_solutions(p1) == 1      # exactly one solution
    sol, _ = solve(p1, use_fc=True, use_ac3=True)
    assert flat(sol) == flat_grid(s1)


def test_normal_5_search_finds_known_fix():
    """The example from the project document: 52% wrong digit vs 45% true digit."""
    puzzle, _ = make_puzzle(1, 30)
    for r in range(9):
        given = [c for c in range(9) if puzzle[r][c]]
        if len(given) >= 2:
            c, other = given[0], given[1]
            break
    true_digit, wrong = puzzle[r][c], puzzle[r][other]
    grid = [row[:] for row in puzzle]
    grid[r][c] = wrong
    probs = np.full((9, 9, 9), 0.001)
    for rr in range(9):
        for cc in range(9):
            if grid[rr][cc]:
                probs[rr, cc, grid[rr][cc] - 1] = 0.98
    probs[r, c, wrong - 1] = 0.52
    probs[r, c, true_digit - 1] = 0.45

    res = correct(grid, probs, "ucs")
    cell, old, new, cost = res["log"][0]
    assert res["found"] and len(res["log"]) == 1
    assert cell == (r, c) and new == true_digit
    assert abs(cost - math.log(0.52 / 0.45)) < 1e-9


# ---------------- 3 EDGE ----------------
def test_edge_1_already_full_grid():
    sol, stats = solve(to_grid(ANSWER), use_fc=True, use_ac3=True)
    assert flat(sol) == ANSWER
    assert stats["assignments"] == 0


def test_edge_2_empty_grid_still_solves():
    sol, _ = solve([[0] * 9 for _ in range(9)], use_fc=True, use_ac3=True)
    assert sol is not None
    for r in range(9):
        assert sorted(sol[(r, c)] for c in range(9)) == list(range(1, 10))


def test_edge_3_correct_grid_needs_no_correction():
    grid = to_grid(PUZZLE)
    probs = np.full((9, 9, 9), 1 / 9)
    res = run_pipeline(grid, probs)
    assert res["status"] == "solved"
    assert res["log"] == []


# ---------------- 2 IMPOSSIBLE ----------------
def test_impossible_1_two_equal_givens_no_solution():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 5
    grid[0][1] = 5                       # same digit twice in one row
    sol, _ = solve(grid, use_fc=True, use_ac3=True)
    assert sol is None
    assert ac3_conflict(grid) is True    # AC-3 itself reports the conflict
    res = run_pipeline(grid, np.full((9, 9, 9), 1 / 9), mode="off")
    assert "no solution" in res["status"]
    print(res["status"])


def test_impossible_2_too_broken_is_unreadable():
    # 4 duplicate pairs in one row need 4 fixes, but Search allows only 3
    row = [1, 1, 2, 2, 3, 3, 4, 4, 5]
    grid = [[0] * 9 for _ in range(9)]
    grid[0] = row
    probs = np.random.RandomState(0).dirichlet(np.ones(9), size=(9, 9))
    res = run_pipeline(grid, probs, mode="nn")
    assert res["status"] == "unreadable"
    assert len(res["least_confident"]) > 0
    print(res["status"], res["least_confident"])
