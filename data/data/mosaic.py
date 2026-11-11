import numpy as np


def load_pools(model, path="data/mnist_split.npz"):
    """Sort test images into 'good' (read correctly) and 'hard' (misread)."""
    d = np.load(path)
    X = d["X_test"] / 255.0
    y = d["y_test"]
    probs = model.predict_proba(X)
    pred = probs.argmax(axis=1) + 1
    true_prob = probs[np.arange(len(y)), y - 1]
    rank = (probs > true_prob[:, None]).sum(axis=1)   # 0 = true digit is the top guess
    good = {k: np.where((pred == y) & (y == k))[0] for k in range(1, 10)}
    # HARD rule: the NN is wrong, but the true digit is in its top 3
    hard = {k: np.where((pred != y) & (rank < 3) & (y == k))[0] for k in range(1, 10)}
    return X, good, hard


def make_mosaic(puzzle, n_misreads, seed, X, good, hard):
    """Return (images (81,784), list of cells that hold hard digits)."""
    rng = np.random.RandomState(seed)
    cells = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c]]
    usable = [rc for rc in cells if len(hard[puzzle[rc[0]][rc[1]]])]
    if n_misreads > len(usable):
        raise ValueError("not enough hard digits for this puzzle")
    pick = rng.choice(len(usable), n_misreads, replace=False)
    bad = [usable[i] for i in pick]
    images = np.zeros((9, 9, 784))                    # blank = all black
    for r, c in cells:
        d = puzzle[r][c]
        pool = hard[d] if (r, c) in bad else good[d]
        images[r, c] = X[rng.choice(pool)]
    return images.reshape(81, 784), bad
