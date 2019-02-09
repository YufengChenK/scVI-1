import pickle
import numpy as np
with open("X.pkl", 'rb') as xf:
    X = pickle.load(xf)
print(X)
X = np.sum(X, axis=0)
print(np.shape(X))
X = np.array(X).ravel()
print(X)
