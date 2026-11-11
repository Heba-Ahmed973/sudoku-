import numpy as np
import joblib

BLANK_THRESHOLD = 0.02   # a blank cell is almost completely black


class Reader:
    def __init__(self, model_path="data/nn.joblib"):
        self.model = joblib.load(model_path)

    def read_grid(self, images):
        """images: (81, 784) numbers 0-1. Returns (9x9 grid, probs 9x9x9)."""
        blank = images.mean(axis=1) < BLANK_THRESHOLD    # brightness rule
        probs = np.zeros((81, 9))
        if (~blank).any():
            probs[~blank] = self.model.predict_proba(images[~blank])
        grid = np.zeros(81, dtype=int)
        grid[~blank] = probs[~blank].argmax(axis=1) + 1
        return grid.reshape(9, 9).tolist(), probs.reshape(9, 9, 9)
