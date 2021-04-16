import numpy as np

def l2norm(points, center, axis=0):
    dist = (points - center)**2
    dist = np.sum(dist, axis=1)
    return np.sqrt(dist)