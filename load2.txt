from .VecLD import VecLD
import numpy as np
import scipy.io

def importMat(self, filename_mat):
    """
    Imports a vectorized linedrawing from a .mat file.

    Args:
        filename_mat (str): the filename of the .mat file to import
    """
    mat = scipy.io.loadmat(filename_mat)

    self.originalImage: str = mat['vecLD']['originalImage'][0][0][0]
    self.imsize: np.ndarray = mat['vecLD']['imsize'][0][0][0]
    self.lineMethod: str = mat['vecLD']['lineMethod'][0][0][0]
    self.numContours: int = int(mat['vecLD']['numContours'][0][0][0])
    self.contours: np.ndarray = mat['vecLD']['contours'][0][0][0]

setattr(VecLD, 'importMat', importMat)


def importMatNew(filename_mat):
    """
    Imports a vectorized linedrawing from a .mat file.

    Args:
        filename_mat (str): the filename of the .mat file to import

    Returns:
        The vectorized linedrawing as a VecLD object
    """
    mat = scipy.io.loadmat(filename_mat)

    originalImage: str = mat['vecLD']['originalImage'][0][0][0]
    imsize: np.ndarray = mat['vecLD']['imsize'][0][0][0]
    lineMethod: str = mat['vecLD']['lineMethod'][0][0][0]
    numContours: int = mat['vecLD']['numContours'][0][0][0]
    contours: np.ndarray = mat['vecLD']['contours'][0][0][0]

    return VecLD(originalImage, imsize, lineMethod, numContours, contours)

setattr(VecLD, 'importMatNew', importMatNew)
