from .VecLD import VecLD
import numpy as np

# @classmethod
def rotateLinedrawing(cls, vecLD, angle):
    """
    Rotate a linedrawing by a given angle (in degrees).

    Args:
        vecLD (VecLD): the vectorized linedrawing to rotate
        angle (float): the angle to rotate the linedrawing by (in degrees)

    Returns:
        The rotated vectorized linedrawing as a VecLD object.
    """
    rotatedContours = np.empty_like(vecLD.contours)
    
    centerPoint = vecLD.imsize[[0, 1, 0, 1]] / 2

    sinAngle = np.sin(np.deg2rad(angle))
    cosAngle = np.cos(np.deg2rad(angle))

    for c in range(vecLD.numContours):
        # Calculate the offset for the contour
        offset = np.tile(centerPoint, (vecLD.contours[c].shape[0], 1))

        # Subtract the offset from the contour
        con = vecLD.contours[c] - offset

        # Rotate the contour
        rot = np.empty_like(con)
        rot[:,0] = cosAngle * con[:,0] - sinAngle * con[:,1]
        rot[:,1] = sinAngle * con[:,0] + cosAngle * con[:,1]
        rot[:,2] = cosAngle * con[:,2] - sinAngle * con[:,3]
        rot[:,3] = sinAngle * con[:,2] + cosAngle * con[:,3]

        # Add the offset back to the rotated contour
        rotatedContours[c] = rot + offset
    
    return cls(
        vecLD.originalImage,
        vecLD.imsize,
        vecLD.lineMethod,
        vecLD.numContours,
        rotatedContours # rotated contours
    )

setattr(VecLD, 'rotateLinedrawing', rotateLinedrawing)
