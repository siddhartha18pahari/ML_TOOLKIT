from .VecLD import VecLD
import numpy as np
import matplotlib.pyplot as plt

def drawLinedrawing(
    self,
    filename: str = 'linedrawing.png',
    lineWidth: float = 1,
    color: np.ndarray = np.array([0, 0, 0]) 
):
    """
    Draws the line drawing and saves it to a file.

    Args:
        filename (str): The name of the file to save the image to.
        lineWidth (float): The width of the lines in the image.
        color (np.ndarray): The color of the lines in the image.
    """
    plt.clf()  # Clear the current figure
    
    for c in range(self.numContours):
        thisC = self.contours[c]
        X = np.concatenate((thisC[:,0], thisC[-1,2]), axis=None)
        Y = np.concatenate((thisC[:,1], thisC[-1,3]), axis=None)
        plt.plot(X, Y, '-', color=color, linewidth=lineWidth)
        
    plt.gca().set_aspect('equal', adjustable='box')
    
    ### for removing ticks
    # plt.gca().set_xticks([])
    # plt.gca().set_yticks([])

    ### sets distance of tick lines from axis
    # plt.gca().tick_params(length=0)

    plt.axis('image')
    plt.axis([1, self.imsize[0], 1, self.imsize[1]])

    plt.gca().invert_yaxis()

    plt.show()
    plt.savefig(filename, bbox_inches='tight')
    
setattr(VecLD, 'drawLinedrawing', drawLinedrawing)
