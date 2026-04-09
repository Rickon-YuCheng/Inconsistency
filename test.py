import numpy as np

PL=np.load("PseudoLabel.npz")
print(PL["a"])
print(type(PL["a"]))