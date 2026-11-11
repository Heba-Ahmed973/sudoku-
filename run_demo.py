import argparse
import matplotlib.pyplot as plt
from csp.generator import make_puzzle
from nn.reader import Reader
from data.mosaic import load_pools, make_mosaic
from integration.pipeline import run_pipeline

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=1)
p.add_argument("--givens", type=int, default=30)
p.add_argument("--misreads", type=int, default=2)
p.add_argument("--method", default="ucs", choices=["ucs", "greedy", "astar"])
p.add_argument("--nn", default="on", choices=["on", "off"])
args = p.parse_args()


def show(grid):
    for row in grid:
        print(" ".join(str(x) if x else "." for x in row))


def draw(ax, grid, title, colors=None):
    ax.set_xlim(0, 9); ax.set_ylim(9, 0); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=10)
    for (r, c), col in (colors or {}).items():
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color=col))
    for i in range(10):
        w = 2.5 if i % 3 == 0 else 0.5
        ax.plot([0, 9], [i, i], "k", lw=w)
        ax.plot([i, i], [0, 9], "k", lw=w)
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                ax.text(c + .5, r + .5, str(grid[r][c]), ha="center", va="center", fontsize=11)


# ---- build the input ----
reader = Reader()
X, good, hard = load_pools(reader.model)
puzzle, _ = make_puzzle(args.seed, args.givens)
images, bad = make_mosaic(puzzle, args.misreads, args.seed, X, good, hard)
grid, probs = reader.read_grid(images)

print(f"INPUT: seed={args.seed}, givens={args.givens}, misreads injected={args.misreads}")
print("Grid as read by the NN:")
show(grid)

# ---- run the system ----
res = run_pipeline(grid, probs, mode="nn" if args.nn == "on" else "off", method=args.method)
print("\nAC-3 found a conflict:", res["ac3_conflict"])
print("Duplicate pairs:", res["conflicts"])
print("RESULT:", res["status"])
if res["log"]:
    print("Correction log (cell, read digit, new digit, cost):")
    for cell, old, new, cost in res["log"]:
        print("  ", cell, old, "->", new, round(cost, 3))
    print("Search counters:", res["stats"])
if res["status"] == "unreadable":
    print("Least confident cells (cell, confidence):", res["least_confident"])
if res["solution"]:
    print("Solved grid:")
    show(res["solution"])
    print("Matches the true puzzle's solution cells:",
          all(res["corrected_grid"][r][c] == puzzle[r][c]
              for r in range(9) for c in range(9)))

# ---- plots ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
picture = images.reshape(9, 9, 28, 28).transpose(0, 2, 1, 3).reshape(252, 252)
axes[0].imshow(picture, cmap="gray"); axes[0].axis("off")
axes[0].set_title("1. Input mosaic")

colors = {}
for a, b in res["conflicts"]:
    colors[a] = colors[b] = "#fff3a0"                      # yellow = AC-3 conflict
for r in range(9):
    for c in range(9):
        if grid[r][c] != puzzle[r][c]:
            colors[(r, c)] = "#ffb3b3"                     # red = real misread (shown for the demo only)
draw(axes[1], grid, "2. NN reading (yellow = conflict, red = misread)", colors)

if res["solution"]:
    fixed = {(r, c): "#a9d4ff" for (r, c), _, _, _ in res["log"]}
    draw(axes[2], res["solution"], "3. CSP solution (blue = corrected by Search)", fixed)
else:
    weak = {rc: "#ffd9a0" for rc, _ in res["least_confident"]}
    draw(axes[2], grid, "3. " + res["status"], weak)
plt.tight_layout()
plt.savefig("data/demo.png", dpi=120)
plt.show()
