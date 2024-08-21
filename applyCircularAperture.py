from .VecLD import VecLD
import numpy as np

def applyCircularAperture(vecLD: VecLD, radius = None):
    """
    Apply circular aperture to the vector line-drawing.

    Args:
        radius (float): Radius of the circular aperture.
    """
    if not isinstance(vecLD, VecLD):
        raise TypeError('The second argument must be of type VecLD.')
    
    if radius is None:
        radius = np.min(vecLD.imsize) / 2

    maskedLD = VecLD(
        originalImage = vecLD.originalImage,
        imsize = vecLD.imsize,
        lineMethod = vecLD.lineMethod,
        numContours = 0,
        contours = np.empty((0, 4))
    )
    
    center = vecLD.imsize / 2

    for c in range(vecLD.numContours):

        A = vecLD.contours[c][:, :2]
        B = vecLD.contours[c][:, 2:4]
        rA = np.sqrt(np.sum((A - center) ** 2, axis=1))
        rB = np.sqrt(np.sum((B - center) ** 2, axis=1))

        prevInside = (rA[0] <= radius)
        currContour = np.empty((0, 4))

        for s in range(vecLD.contours[c].shape[0]):
            currInside = (rB[s] <= radius)

            # if end points are on different sides, compute the intersection point with the circle
            if (currInside ^ prevInside): # currInside XOR prevInside
                # length of this segment
                d = np.sqrt(np.sum((B[s,:] - A[s,:])**2))

                # solve the quadratic equation
                p = -d - (rA[s]**2 - rB[s]**2) / d
                q = rA[s]**2 - radius**2
                QQ = np.sqrt((p/2)**2 - q)
                dA1 = -p/2 + QQ
                dA2 = -p/2 - QQ

                # make sure we pick the right solution
                dA1valid = (0 <= dA1) & (dA1 <= d)
                dA2valid = (0 <= dA2) & (dA2 <= d)
                if dA1valid:
                    dA = dA1
                    if dA2valid:
                        raise ValueError("Two valid solution - don't know which one to pick.")
                elif dA2valid:
                    dA = dA2
                else:
                    raise ValueError("No valid solution - don't know what to do.")
                
                C = A[s,:] + dA/d * (B[s,:]-A[s,:])

            # consider all 4 cases
            if prevInside:
                if currInside:
                    # we are completely inside the circle - just keep the segment
                    currContour = np.vstack((currContour, vecLD.contours[c][s,:]))
                else:
                    # going from inside to outside the circle
                    # break the segment and terminate this contour
                    currContour = np.vstack((currContour, np.hstack((A[s,:], C))))
                    maskedLD.numContours += 1
                    maskedLD.contours = np.concatenate((maskedLD.contours, [currContour]))
                    currContour = np.empty((0, 4))
            else:
                if currInside:
                    # going from outside to inside
                    # break the segment and start a new contour
                    currContour = np.vstack((C, B[s,:]))
                    maskedLD.numContours += 1
                    maskedLD.contours = np.concatenate((maskedLD.contours, [currContour]))
                    currContour = np.empty((0, 4))
                else:
                    # completely outside - do nothing
                    pass

    prevInside = currInside
    
    # save the contour if it is non-empty
    if currContour.size != 0:
        maskedLD.numContours += 1
        maskedLD.contours = np.concatenate((maskedLD.contours, [currContour]))
    
    return maskedLD

setattr(VecLD, 'applyCircularAperture', applyCircularAperture)
