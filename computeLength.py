from .VecLD import VecLD
import numpy as np

def computeLength(self):
    """
    Computes the length for the contours in the vectorized line drawing `vecLD`.

    Args:
        vecLD (dict): A dictionary representing the vectorized line drawing data structure that has length information added via the function.
    """
    vecLD.lengths = []
    vecLD.contourLengths = np.zeros((vecLD.numContours, 1))
    for c in range(vecLD.numContours):
        thisCon = vecLD.contours[c]
        x_diff = thisCon[:, 2] - thisCon[:, 0]
        y_diff = thisCon[:, 3] - thisCon[:, 1]
        lengths = np.sqrt(x_diff**2 + y_diff**2)
        vecLD.lengths.append(lengths)
        vecLD.contourLengths[c, 0] = np.sum(lengths)

setattr(VecLD, 'computeLength', computeLength)
