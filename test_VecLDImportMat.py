from MLV_toolbox.core import VecLD, importMat
import pytest
import numpy as np

@pytest.fixture
def get_test_VecLDImportMat_matfile():
    return 'test_VecLDImportMat.mat'

@pytest.fixture
def get_test_VecLDImportMat_data():
    return [(
        'shapes.jpg',
        np.array([200, 200]),
        'traceLineDrawingFromRGB',
        51,
        'test_VecLDImportMat.npy'
    )]

def test_VecLDmImportMat(get_test_VecLDImportMat_data):
    for data in get_test_VecLDImportMat_data:
        originalImage = data[0]
        imsize = data[1]
        lineMethod = data[2]
        numContours = data[3]
        contours = None
        with open(data[4], 'rb') as f:
            contours = np.load(f, allow_pickle=True)
        
    expected = VecLD(originalImage, imsize, lineMethod, numContours, contours)

    loaded = VecLD()
    loaded.importMat(get_test_VecLDImportMat_matfile())
    
    print(loaded)
    print(expected)

    assert loaded == expected
