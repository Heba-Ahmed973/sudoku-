import numpy as np
from sklearn.datasets import fetch_openml

X, y = fetch_openml("mnist_784", version=1, as_frame=False,
                    return_X_y=True, parser="liac-arff")
y = y.astype(int)

# Same split and seed as train.py
idx = np.random.RandomState(0).permutation(60000)
X_tr, y_tr = X[:60000][idx], y[:60000][idx]
X_val, y_val = X_tr[50000:], y_tr[50000:]
X_te, y_te = X[60000:], y[60000:]

k = y_val != 0
X_val, y_val = X_val[k], y_val[k]
k = y_te != 0
X_te, y_te = X_te[k], y_te[k]

np.savez_compressed("data/mnist_split.npz",
                    X_val=X_val.astype(np.uint8), y_val=y_val,
                    X_test=X_te.astype(np.uint8), y_test=y_te)
print("Saved data/mnist_split.npz")
