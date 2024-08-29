from .VecLD import VecLD
import numpy as np

def lineIntersection(
    queryLine: np.ndarray,
    refLine: np.ndarray,
    RE: float = 0.3,
    AE: float = 2,
) -> np.ndarray:
    """
    Determine if two line segments intersect and, if so, where.

    Args:
        queryLine (np.ndarray): A 2x2 array representing the query line segment
            with start and end coordinates: [X1,Y1,X2,Y2].
        refLine (np.ndarray): A 2x2 array representing the reference line
            segment with start and end coordinates: [X1,Y1,X2,Y2].
        RE (float): The relative error threshold for the intersection point.
            Default is 0.3.
        AE (float): The absolute error threshold for the intersection point in
            pixels. Default is 2.

    Returns:
        Position (np.ndarray): A 2x1 array representing the intersection point
            with coordinates: [X,Y]. If the lines do not intersect, Position
            will be empty [].
    """
    
    eps = 1e-4
    
    Ay = queryLine[2] - queryLine[0]
    Ax = queryLine[3] - queryLine[1]
    By = refLine[2] - refLine[0]
    Bx = refLine[3] - refLine[1]
    Cy = refLine[0] - queryLine[0]
    Cx = refLine[1] - queryLine[1]
    
    D = Ay * Bx - Ax * By
    
    np.seterr(divide='ignore', invalid='ignore') # divide by zero is okay here
    a = (Bx * Cy - By * Cx) / D
    b = (Ax * Cy - Ay * Cx) / D
    
    # calculate ratio thresholds
    at = min(RE, AE / np.max(np.abs([Ax, Ay])))
    bt = min(RE, AE / np.max(np.abs([Bx, By])))
    
    if (-at <= a) and (a <= 1 + at) and (-bt <= b) and (b <= 1 + bt):
        # special cases where a or b are 0 or 1
        if np.abs(a) < eps:
            Position = queryLine[0:2]
        elif np.abs(a - 1) < eps:
            Position = queryLine[2:4]
        elif np.abs(b) < eps:
            Position = refLine[0:2]
        elif np.abs(b - 1) < eps:
            Position = refLine[2:4]
        else:
            # general case
            A1 = queryLine[1] - queryLine[3]
            B1 = queryLine[2] - queryLine[0]
            C1 = queryLine[0] * queryLine[3] - queryLine[1] * queryLine[2]
        
            A2 = refLine[1] - refLine[3]
            B2 = refLine[2] - refLine[0]
            C2 = refLine[0] * refLine[3] - refLine[1] * refLine[2]
        
            D = A1 * B2 - A2 * B1
        
            Position = np.zeros(2)
            Position[0] = (B1 * C2 - B2 * C1) / D
            Position[1] = (A2 * C1 - A1 * C2) / D
            
            ##### ALTERNATIVELY:
            # general case
            """ A = np.array([[queryLine[1] - queryLine[3], queryLine[2] - queryLine[0]],
                        [refLine[1] - refLine[3], refLine[2] - refLine[0]]])
            B = np.array([queryLine[0] * queryLine[3] - queryLine[1] * queryLine[2],
                        refLine[0] * refLine[3] - refLine[1] * refLine[2]])
            Position = np.linalg.solve(A, B) """
            
            if np.any(Position < 0):
                Position = []
    else:
        Position = []
        
setattr(VecLD, 'lineIntersection', lineIntersection)
