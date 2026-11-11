from csp.generator import make_puzzle
from nn.reader import Reader
from data.mosaic import load_pools, make_mosaic
from search.correct import correct

reader = Reader()
X, good, hard = load_pools(reader.model)
puzzle, _ = make_puzzle(seed=1, givens=30)
images, bad = make_mosaic(puzzle, 2, 1, X, good, hard)
grid, probs = reader.read_grid(images)

for method in ["ucs", "greedy", "astar"]:
    res = correct(grid, probs, method)
    print("==", method, "==")
    print("found:", res["found"], res["stats"])
    for cell, old, new, cost in res["log"]:
        print("  cell", cell, "read", old, "->", new, "cost", round(cost, 3))
    if res["found"]:
        print("  exact recovery:", res["grid"] == puzzle)
