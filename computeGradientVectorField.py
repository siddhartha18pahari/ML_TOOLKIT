from .VecLD import VecLD
import numpy as np
from typing import Tuple
from scipy.ndimage.morphology import distance_transform_edt

def is_outer_border_point(
    binaryImage: np.ndarray,
    ii: float,
    jj: float,
    m_Neighbors8: np.ndarray,
    background: int
) -> int:
    """
    Determines whether a point is an outer border point.
    
    Args:
        binaryImage (np.ndarray): A binary image.
        ii (float): The row index of the point.
        jj (float): The column index of the point.
        m_Neighbors8 (np.ndarray): An array of the 8-neighborhood.
        background (int): The background value of the binary image.
        
    Returns:
        result2 (int): 1 if the point is an outer boundary point, 0 otherwise.
    """
    
    if binaryImage[ii, jj] == background:
        result2 = 0
        nOfBackgroundPoints = 0
        nOfForegoundPoints = 0
        iterator = 1
        
        while (nOfBackgroundPoints == 0 or nOfForegroundPoints == 0) and iterator < 8:
            if binaryImage[ii+m_Neighbors8[iterator, 0], jj+m_Neighbors8[iterator, 1]] > background:
                nOfForegroundPoints += 1
            if binaryImage[ii+m_Neighbors8[iterator, 0], jj+m_Neighbors8[iterator, 1]] <= background:
                nOfBackgroundPoints += 1
            iterator += 1
        if nOfBackgroundPoints > 0 and nOfForegroundPoints > 0:
            result2 = 1
    else:
        result2 = 0
        
    return result2

setattr(VecLD, 'is_outer_border_point', is_outer_border_point)




def getOuterBoundary(
    binaryImage: np.ndarray,
    background 
) -> Tuple[
    np.ndarray,
    np.ndarray
]:
    """
    Gets the outer boundary of a binary image.
    
    Args:
        binaryImage (np.ndarray): A binary image.
        background (int): The background value of the binary image.
        
    Returns:
        result (np.ndarray): Stores coordinates of outer boundary points of the 
            binary image.
        result2 (np.ndarray): Binary image with outer boundary points set to 1 
            and all other points set to 0.
    """
    
    outerBoundary = np.empty(())
    m_Neighbors8 = np.array([[-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1], [1,0], [1,1]])
    result2 = np.zeros(binaryImage.shape)
    
    m, n = binaryImage.shape
    result = np.zeros((m*n, 2))
    
    counter = 0
    result = np.zeros((m*n, 2), dtype=int)
    result2 = np.zeros((m, n), dtype=int)

    for i in range(1, m-1):
        for j in range(1, n-1):
            if is_outer_border_point(binaryImage, i, j, m_Neighbors8, background):
                result[counter] = [i, j]
                result2[i, j] = 1
                counter += 1

    result = result[:counter]
    
    return result, result2

setattr(VecLD, 'getOuterBoundary', getOuterBoundary)




def computeGradientVectorField(
    binaryImage: np.ndarray
) -> Tuple[
    np.ndarray, 
    int
]:
    """
    Computes the gradient vector field of a binary image.
    
    Args:
        binaryImage (np.ndarray): A binary image.
        
    Returns:
        D (np.ndarray): distance map computed with respect to the binary image.
        IDX (int): index of closest point to the boundary.
    """
    
    newBinaryImage = binaryImage.copy()
    outerBoundary = getOuterBoundary(binaryImage, 0)
    
    for i in range(outerBoundary.shape[0]):
        newBinaryImage[outerBoundary[i, 0], outerBoundary[i, 1]] = 1
        
    D2, IDX2 = distance_transform_edt(newBinaryImage == 0, return_indices=True)
    D1, IDX1 = distance_transform_edt(binaryImage == 1, return_indices=True)
    
    IDX1[D1 == 0] = 0
    IDX2[D2 == 0] = 0
    
    IDX = IDX1 + IDX2
    for i in range(outerBoundary.shape[0]):
        IDX[outerBoundary[i, 0], outerBoundary[i, 1]] = np.ravel_multi_index((outerBoundary[i, 0], outerBoundary[i, 1]), IDX.shape)

    D = D1 - D2
    
    return D, IDX

setattr(VecLD, 'computeGradientVectorField', computeGradientVectorField)
