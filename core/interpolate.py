import numpy as np

def quaficient(a, d, n=3):
    a_np = np.array(a)
    d_np = np.array(d)
    return np.linalg.solve(a_np, d_np)
