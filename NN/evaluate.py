import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix

# Load MNIST with the SAME split as train.py
X, y = fetch_openml("mnist_784", version=1, as_frame=False,
                    return_X_y=True, parser="liac-arff")
y = y.astype(int)
X = X / 255.0
idx = np.random.RandomState(0).permutation(60000)
X_tr, y_tr = X[:60000][idx], y[:60000][idx]
X_val, y_val = X_tr[50000:], y_tr[50000:]
X_te, y_te = X[60000:], y[60000:]
k = y_val != 0
X_val, y_val = X_val[k], y_val[k]
k = y_te != 0
X_te, y_te = X_te[k], y_te[k]


def ece(probs, y, model, bins=10):
    """Calibration error: do 90% confident guesses get 90% right?"""
    conf = probs.max(axis=1)
    pred = model.classes_[probs.argmax(axis=1)]
    right = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(right[m].mean() - conf[m].mean())
    return total


for name in ["nn", "lr"]:
    model = joblib.load(f"data/{name}.joblib")
    probs = model.predict_proba(X_te)
    pred = model.classes_[probs.argmax(axis=1)]
    print("====", name.upper(), "(TEST set) ====")
    print("accuracy:", (pred == y_te).mean())
    print("calibration error (ECE):", ece(probs, y_te, model))
    if name == "nn":
        print("confusion matrix (rows = true digit 1-9):")
        print(confusion_matrix(y_te, pred))
        top = probs.max(axis=1)
        print("test images with top probability < 0.6:", int((top < 0.6).sum()))
