from .VecLD import VecLD
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from typing import Tuple

def computeColorIndex(
    vecLD: VecLD,
    property: str
) -> Tuple[
    np.ndarray, 
    np.ndarray
]:
    """
    Computes the color index for a given property of a vector of contours.

    Args:
        vecLD: A vector of contours represented as a VecLD object.
        property: A string indicating the property to use for computing the color index.
            Possible values are 'length', 'curvature', and 'orientation'.

    Returns:
        A tuple containing the color index as a numpy array and the colormap as a matplotlib colormap object.

    Raises:
        ValueError: If the specified property is not one of 'length', 'curvature', or 'orientation'.
    """
    
    property = property.lower()
    
    colorIdx = np.array([])
    numCols = 256
    
    if property == 'length':
        allLengths = np.log10(vecLD.contourLengths+1)
        
        minProp = np.min(allLengths)
        maxProp = np.max(allLengths)
        
        col = np.round((allLengths - minProp) / (maxProp-minProp) * (numCols-1) + 1)
        
        for c in range(vecLD.numContours):
            colorIdx.append(np.zeros((vecLD.contours[c].shape[0], 1)) + col[c])
        
        cmap = plt.get_cmap('jet', numCols)

    elif property == 'curvature':
        allCurv = np.log10([item for sublist in vecLD.curvatures for item in sublist]+1)
        
        maxProp = np.max(allCurv)*0.8
        minProp = np.min(allCurv)
        
        for c in range(vecLD.numContours):
            colorIdx.append(np.minimum(np.round((np.log10(vecLD.curvatures[c]+1) - minProp) / (maxProp-minProp) * (numCols-1) + 1), numCols))
        
        cmap = plt.get_cmap('jet', numCols)

    elif property == 'orientation':
        for c in range(vecLD.numContours):
            colorIdx.append(np.round(np.mod(vecLD.orientations[c], 180) / 180 * (numCols-1) + 1))
        
        cmap = plt.get_cmap('hsv', numCols)

    else:
        raise ValueError(f'Unknown property: {property}')

    return colorIdx, cmap

setattr(VecLD, 'computeColorIndex', computeColorIndex)
