import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# 1. Download MNIST (first time only, needs internet)
X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True, parser="liac-arff")
y = y.astype(int)
X = X / 255.0

# 2. Shuffle the 60000 training images (fixed seed), THEN split 50000 / 10000
idx = np.random.RandomState(0).permutation(60000)
X_tr, y_tr = X[:60000][idx], y[:60000][idx]
X_train, y_train = X_tr[:50000], y_tr[:50000]
X_val, y_val = X_tr[50000:], y_tr[50000:]
X_te, y_te = X[60000:], y[60000:]   # test set, don't touch yet

# 3. Drop digit 0 from all three sets
k = y_train != 0
X_train, y_train = X_train[k], y_train[k]
k = y_val != 0
X_val, y_val = X_val[k], y_val[k]
k = y_te != 0
X_te, y_te = X_te[k], y_te[k]

# 4. Train the neural network
nn = MLPClassifier(hidden_layer_sizes=(128,), max_iter=20, random_state=0)
nn.fit(X_train, y_train)
print("NN validation accuracy:", nn.score(X_val, y_val))

# 5. Train the baseline
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
print("Logistic regression validation accuracy:", lr.score(X_val, y_val))

# 6. Save both models
joblib.dump(nn, "data/nn.joblib")
joblib.dump(lr, "data/lr.joblib")
print("Saved.")