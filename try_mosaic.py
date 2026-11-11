import matplotlib.pyplot as plt
from csp.generator import make_puzzle
from nn.reader import Reader
from data.mosaic import load_pools, make_mosaic

reader = Reader()
X, good, hard = load_pools(reader.model)
print("hard images available:", sum(len(v) for v in hard.values()))

puzzle, solution = make_puzzle(seed=1, givens=30)
images, bad = make_mosaic(puzzle, 2, 1, X, good, hard)
grid, probs = reader.read_grid(images)

print("cells I made hard:", bad)
for r in range(9):
    for c in range(9):
        if grid[r][c] != puzzle[r][c]:
            print("misread at", (r, c), "real:", puzzle[r][c], "read:", grid[r][c])

picture = images.reshape(9, 9, 28, 28).transpose(0, 2, 1, 3).reshape(252, 252)
plt.imshow(picture, cmap="gray")
plt.axis("off")
plt.savefig("data/mosaic.png")
plt.show()
