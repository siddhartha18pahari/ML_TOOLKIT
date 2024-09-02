from .VecLD import VecLD
import numpy as np

def randomlyShiftContours(
    vecLD: VecLD,
    maxShift: int,
) -> VecLD:
    """
    Randomly shifts the contours within the image.

    Args:
        vecLD (VecLD): A vectorized line drawing.
        maxShift (int): A scalar for the maximum number of pixels used for the shift
            or a two-element vector specifying the maximum shift in the
            x and y direction.

    Returns:
        A new vectorized line drawing with the shifted contours.
    """
    
    if maxShift is None:
        maxShift = vecLD.imsize
    if isinstance(maxShift, int):
        maxShift = (maxShift, maxShift)
    
    shiftedLD = VecLD(
        originalImage = vecLD.originalImage,
        imsize = vecLD.imsize,
        lineMethod = vecLD.lineMethod,
        numContours = vecLD.numContours,
        contours = np.empty_like(vecLD.contours)
    )
    
    for c in range(vecLD.numContours):
        # X direction
        minX = np.min(np.concatenate((vecLD.contours[c][:, 0], vecLD.contours[c][:, 2])))
        maxX = np.max(np.concatenate((vecLD.contours[c][:, 0], vecLD.contours[c][:, 2])))
        
        lowX = min(minX-1, maxShift[0])
        highX = min(vecLD.imsize[0] - maxX, maxShift[0])
        shiftX = np.random.randint(-lowX, highX+1)
        
        # Y direction
        minY = np.min(np.concatenate((vecLD.contours[c][:, 1], vecLD.contours[c][:, 3])))
        maxY = np.max(np.concatenate((vecLD.contours[c][:, 1], vecLD.contours[c][:, 3])))

        lowY = min(minY-1, maxShift[1])
        highY = min(vecLD.imsize[1] - maxY, maxShift[1])
        shiftY = np.random.randint(-lowY, highY+1)
        
        # shift the coordinates
        shiftVector = np.array([shiftX, shiftY, shiftX, shiftY])
        shiftedLD.contours[c] = vecLD.contours[c] + np.tile(shiftVector, (vecLD.contours[c].shape[0], 1))
    
    return shiftedLD

setattr(VecLD, 'randomlyShiftContours', randomlyShiftContours)
