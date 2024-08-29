
from .VecLD import VecLD
import numpy as np
import xml.etree.ElementTree as xmltree

def getAttribute(theNode, attrName):
    """
    Returns the value of the specified attribute for the given XML node.

    Args:
        theNode (xml.etree.ElementTree.Element): The XML node to retrieve the attribute from.
        attrName (str): The name of the attribute to retrieve.

    Returns:
        str: The value of the specified attribute, or None if the attribute is not found.
    """
    attribute = None
    if theNode.hasAttributes():
        theAttributes = theNode.attributes
        numAttributes = theAttributes.length
        for a in range(numAttributes):
            thisAttr = theAttributes.item(a)
            if thisAttr.name == attrName:
                attribute = thisAttr.value
                break
    return attribute



def getValue(theNode, attrName):
    """
    Returns the value of an attribute with the specified name from an XML node as a float.

    Args:
        theNode (xml.dom.minidom.Element): A DOM node representing an element in an XML document.
        attrName (str): The name of the attribute to read.

    Returns:
        float: The value of the attribute as a float, or None if the attribute is not found or cannot be converted to a float.

    """
    attrString = getAttribute(theNode, attrName)
    if not attrString:
        return None
    try:
        attrValue = float(attrString)
    except ValueError:
        attrValue = None
    return attrValue



def parseChildNodes(theNode, vecLD, groupTransform):
    """
    Traverses the SVG tree and fills in the vecLD data structure along the way.

    Args:
        theNode (xml.dom.minidom.Element): A DOM node representing an element in an XML document.
        vecLD (numpy.ndarray): A NumPy structured array for storing information about vector line detection.
        groupTransform (numpy.ndarray): A 3x3 transformation matrix for transforming the coordinates of vector lines in the SVG file.

    Returns:
        numpy.ndarray: The updated vecLD structured array.
    """
    
    name = theNode.tag
    # print(f'Node name: {name}\n');
    if name:
        thisContour = np.empty((0, 2))
        contourBreaks = [1]
        
        if name == 'g':
            thisTransform = getAttribute(theNode, 'transform')
            if thisTransform:
                if not groupTransform:
                    groupTransform = thisTransform
                else:
                    groupTransform = groupTransform + ' ' + thisTransform
                    
        elif name == 'line':
            thisContour = np.zeros(4)
            thisContour[0] = getValue(theNode, 'x1')
            thisContour[1] = getValue(theNode, 'y1')
            thisContour[2] = getValue(theNode, 'x2')
            thisContour[3] = getValue(theNode, 'y2')

        elif name in ['polygon', 'polyline']:
            points = getValue(theNode, 'points')
            x = points[::2]
            y = points[1::2]

            # if polygon isn't closed, close it
            if name == 'polygon' and (x[0] != x[-1] or y[0] != y[-1]):
                x = np.append(x, x[0])
                y = np.append(y, y[0])
            
            thisContour = np.column_stack((x[:-1], y[:-1], x[1:], y[1:]))

        elif name == 'rect':
            x = getValue(theNode, 'x')
            y = getValue(theNode, 'y')
            w = getValue(theNode, 'width')
            h = getValue(theNode, 'height')
            thisContour = np.array([[x, y, x+w, y], [x+w, y, x+w, y+h], [x+w, y+h, x, y+h], [x, y+h, x, y]])
            
        elif name in ['circle', 'ellipse']:
            cx = getValue(theNode, 'cx')
            cy = getValue(theNode, 'cy')
            if name == 'circle':
                rx = abs(getValue(theNode, 'r'))
                ry = rx
            else:
                rx = abs(getValue(theNode, 'rx'))
                ry = abs(getValue(theNode, 'ry'))
            numSeg = max(8, round(2 * np.pi * max(rx, ry) / 5))
            dAng = 360 / numSeg
            angles = np.arange(0, 360, dAng)
            x = rx * np.cos(np.deg2rad(angles)) + cx
            y = ry * np.sin(np.deg2rad(angles)) + cy
            thisContour = np.column_stack((x[:-1], y[:-1], x[1:], y[1:]))
        
        elif name == 'path':
            commands = getAttribute(theNode, 'd')
            commands = commands.replace(',', ' ')
            idx = 1
            prevPos = []
            pathStartPos = []
            prevContr = []
            prevCom = ''
            nextCom = ''
            contourBreaks = []
            
            while (idx <= len(commands)) or (nextCom != ''):
                if nextCom == '':
                    # read the command and coordinates from the command string
                    thisCom = commands[idx]
                    coords = np.fromstring(commands[idx+1:], sep=' ')
                    numCoords = len(coords)
                    if numCoords % 2 != 0:
                        numCoords -= 1
                    coords = coords[:numCoords].reshape(-1, 2)
                    idx += numCoords + 1
                else:
                    thisCom = nextCom
                    nextCom = ''
                # print(f'\tPath command: {thisCom}\n');
            
                # Move pen without drawing - lower case means relative coordinates
                if thisCom in ['M', 'm']:
                    x = coords[::2]
                    y = coords[1::2]

                    contourBreaks.append(thisContour.shape[0] + 1)

                    # relative coords? cumulative addition of points
                    if thisCom == 'm':
                        if prevPos is not None:
                            x[0] += prevPos[0]
                            y[0] += prevPos[1]
                        x = np.cumsum(x)
                        y = np.cumsum(y)

                    # add straight line segments if we have more than one point
                    if len(x) > 1:
                        newContour = np.column_stack((x[:-1], y[:-1], x[1:], y[1:]))
                        thisContour = np.vstack((thisContour, newContour))

                    prevPos = np.array([x[-1], y[-1]])
                    pathStartPos = np.array([x[0], y[0]])
                    
                # draw sequence of line segments
                elif thisCom in ['L', 'l']:
                    x = coords[::2]
                    y = coords[1::2]

                    # connect to previous point
                    x = np.concatenate(([prevPos[0]], x))
                    y = np.concatenate(([prevPos[1]], y))

                    # relative coords? cumulative addition of points
                    if thisCom == 'l':
                        x = np.cumsum(x)
                        y = np.cumsum(y)

                    # add straight line segments
                    thisContour = np.concatenate((thisContour, np.column_stack((x[:-1], y[:-1], x[1:], y[1:]))))
                    prevPos = np.array([x[-1], y[-1]])
            
                # draw horizontal line(s)
                elif thisCom in ['H', 'h']:
                    x = np.concatenate(([prevPos[0]], coords))
                    y = prevPos[1] + np.zeros_like(x)

                    if thisCom == 'h':
                        x = np.cumsum(x)

                    newContour = np.column_stack((x[:-1], y[:-1], x[1:], y[1:]))
                    thisContour = np.vstack((thisContour, newContour))

                    prevPos = np.array([x[-1], y[-1]])
                
                # draw vertical line(s)
                elif thisCom in ['V', 'v']:
                    y = np.concatenate(([prevPos[1]], coords))
                    x = prevPos[0] + np.zeros_like(y)

                    if thisCom == 'v':
                        y = np.cumsum(y)

                    newContour = np.column_stack((x[:-1], y[:-1], x[1:], y[1:]))
                    thisContour = np.vstack((thisContour, newContour))

                    prevPos = np.array([x[-1], y[-1]])
                
                # quadratic Bezier curves
                elif thisCom in ['Q', 'q', 'T', 't']:
                    P0 = prevPos
                    if thisCom in ['Q', 'q']:
                        numCoord = 4
                        if thisCom == 'Q':
                            P1 = coords[:2]
                            P2 = coords[2:]
                        else:
                            P1 = coords[:2] + P0
                            P2 = coords[2:] + P0
                    else:
                        numCoord = 2
                        if prevCom in ['Q', 'q', 'T', 't']:
                            P1 = 2 * P0 - prevContr
                        else:
                            P1 = P0
                        if thisCom == 'T':
                            P2 = coords[:2]
                        else:
                            P2 = coords[:2] + P0

                    dist = np.linalg.norm(P1 - P0) + np.linalg.norm(P2 - P1)
                    numSeg = max(4, round(dist / 5))
                    t = np.linspace(0, 1, numSeg+1)[:-1]
                    x = (1-t)**2 * P0[0] + 2 * (1-t) * t * P1[0] + t**2 * P2[0]
                    y = (1-t)**2 * P0[1] + 2 * (1-t) * t * P1[1] + t**2 * P2[1]

                    newContour = np.column_stack((x[:-1], y[:-1], x[1:], y[1:]))
                    thisContour = np.vstack((thisContour, newContour))

                    prevPos = P2
                    prevContr = P1
                    if len(coords) > numCoord:
                        coords = coords[numCoord:]
                        nextCom = thisCom
                
                # cubic Bezier curves
                elif thisCom in ['C', 'c', 'S', 's']:
                    P0 = prevPos
                    switch = {
                        'C': (6, coords[:2], coords[2:4], coords[4:6]),
                        'c': (6, coords[:2] + P0, coords[2:4] + P0, coords[4:6] + P0),
                        'S': (4, 2 * P0 - prevContr, coords[:2], coords[2:4]),
                        's': (4, 2 * P0 - prevContr, coords[:2] + P0, coords[2:4] + P0)
                    }
                    numCoord, P1, P2, P3 = switch[thisCom]

                    dist = np.linalg.norm(P1 - P0) + np.linalg.norm(P2 - P1) + np.linalg.norm(P3 - P2)
                    numSeg = max(4, round(dist / 5))
                    t = np.linspace(0, 1, numSeg+1)[:-1]
                    x = (1-t)**3 * P0[0] + 3 * (1-t)**2 * t * P1[0] + 3 * (1-t) * t**2 * P2[0] + t**3 * P3[0]
                    y = (1-t)**3 * P0[1] + 3 * (1-t)**2 * t * P1[1] + 3 * (1-t) * t**2 * P2[1] + t**3 * P3[1]

                    newContour = np.column_stack((x[:-1], y[:-1], x[1:], y[1:]))
                    thisContour = np.vstack((thisContour, newContour))

                    prevPos = P3
                    prevContr = P2
                    if len(coords) > numCoord:
                        coords = coords[numCoord:]
                        nextCom = thisCom
                
                # Arcs
                elif thisCom in ['A', 'a']:
                    numCoord = 7
                    P0 = prevPos
                    rx = abs(coords[0])
                    rx2 = rx**2
                    ry = abs(coords[1])
                    ry2 = ry**2
                    rotAng = coords[2]
                    fA = coords[3] # use large arc?
                    fS = coords[4] # sweep clockwise?
                    if thisCom == 'A':
                        P1 = coords[5:7]
                    else:
                        P1 = coords[5:7] + P0

                    # math for conversion to center parametrization
                    # from: https://www.w3.org/TR/SVG/implnote.html
                    
                    # rotation
                    cosA = np.cos(np.deg2rad(rotAng))
                    sinA = np.sin(np.deg2rad(rotAng))
                    rotMat = np.array([[cosA, sinA], [-sinA, cosA]])

                    P0r = rotMat @ (P0 - P1) / 2
                    P0r2 = P0r * P0r
                    
                    # This is the center of the ellipse in transformed coordinates
                    Cr = np.sqrt((rx2 * ry2 - rx2 * P0r2[1] - ry2 * P0r2[0]) / (rx2 * P0r2[1] + ry2 * P0r2[0])) * np.array([rx * P0r[1] / ry, -ry * P0r[0] / rx])

                    if fA == fS:
                        Cr = -Cr

                    ang0 = np.rad2deg(np.arctan2((P0r[1] - Cr[1]) / ry, (P0r[0] - Cr[0]) / rx))
                    ang1 = np.rad2deg(np.arctan2((-P0r[1] - Cr[1]) / ry, (-P0r[0] - Cr[0]) / rx))

                    if fS:
                        if ang1 < ang0:
                            ang1 += 360
                    else:
                        if ang0 < ang1:
                            ang0 += 360
                            
                    # draw the arc in transformed coordinate space
                    numSeg = max(4, round(2 * np.pi * max(rx, ry) * (ang1 - ang0) / 360 / 5))
                    dAng = (ang1 - ang0) / numSeg
                    angles = np.deg2rad(np.arange(ang0, ang1 + dAng, dAng))
                    xR = rx * np.cos(angles) + Cr[0]
                    yR = ry * np.sin(angles) + Cr[1]
                    
                    # transform back to original space
                    x = cosA * xR - sinA * yR + (P0[0] + P1[0]) / 2
                    y = sinA * xR + cosA * yR + (P0[1] + P1[1]) / 2
                    
                    newContour = np.column_stack((x[:-1], y[:-1], x[1:], y[1:]))
                    thisContour = np.vstack((thisContour, newContour))

                    prevPos = P1
                    if len(coords) > numCoord:
                        coords = coords[numCoord:]
                        nextCom = thisCom
                        
                # close path with a straight line if it isn't closed already
                elif thisCom in ['Z', 'z']:
                    if pathStartPos[0] != prevPos[0] or pathStartPos[1] != prevPos[1]:
                        newContour = np.array([prevPos, pathStartPos])
                        thisContour = np.vstack((thisContour, newContour))
                    prevPos = thisContour[-1, 2:4]
                    pathStartPos = None
                
                else:
                    raise ValueError(f'Unknown path command: {thisCom}')
            
                prevCom = thisCom
        
        # after case 'path'
        elif name in ['text', 'tspan', 'textPath']:
            print('Importing text is not implemented. Please convert text to paths in a graphics program, such as Inkscape or Illustrator.')

        elif name == 'image':
            print('Importing embedded images is not implemented.')

        elif name in ['#document', 'defs', 'style', '#text', '#comment']:
            pass
        
        else:
            print(f'Ignoring element <{name}>')        
            
        if thisContour.size > 0:

            # any transformations?
            transCommand = getAttribute(theNode, 'transform')
            if transCommand == '':
                transCommand = groupTransform
            elif groupTransform != '':
                transCommand = groupTransform + ' ' + transCommand
            
            if transCommand:
                openBrackets = np.where(transCommand == '(')[0]
                closeBrackets = np.where(transCommand == ')')[0]
                numTransforms = len(openBrackets)
                closeBrackets = np.concatenate(([-1], closeBrackets))

                for t in range(numTransforms - 1, -1, -1):
                    thisCommand = transCommand[closeBrackets[t]+2:openBrackets[t]-1]
                    #print(f'Transformation: {thisCommand}')
                    valStr = transCommand[openBrackets[t]+1:closeBrackets[t+1]-1]
                    valStr = valStr.replace(',', ' ')
                    values = np.fromstring(valStr, sep=' ', dtype=float)
                    
                    if thisCommand == 'scale':
                        if len(values) == 1:
                            thisContour *= values
                        else:
                            thisContour[:, [0, 2]] *= values[0]
                            thisContour[:, [1, 3]] *= values[1]
                            
                    elif thisCommand == 'translate':
                        if len(values) == 1:
                            values = np.array([values[0], 0])
                        thisContour[:, [0, 2]] += values[0]
                        thisContour[:, [1, 3]] += values[1]
                        
                    elif thisCommand == 'rotate':
                        cc = thisContour.copy()
                        if len(values) == 3:
                            cc[:, [0, 2]] += values[1]
                            cc[:, [1, 3]] += values[2]
                        thisContour[:, [0, 2]] = np.cos(np.deg2rad(values[0])) * cc[:, [0, 2]] - np.sin(np.deg2rad(values[0])) * cc[:, [1, 3]]
                        thisContour[:, [1, 3]] = np.sin(np.deg2rad(values[0])) * cc[:, [0, 2]] + np.cos(np.deg2rad(values[0])) * cc[:, [1, 3]]
                        if len(values) == 3:
                            thisContour[:, [0, 2]] -= values[1]
                            thisContour[:, [1, 3]] -= values[2]
                            
                    elif thisCommand == 'skewX':
                        thisContour[:, [1, 3]] += np.tan(np.deg2rad(values[0])) * thisContour[:, [0, 2]]
                        
                    elif thisCommand == 'matrix':
                        cc = thisContour.copy()
                        thisContour[:, [0, 2]] = values[0] * cc[:, [0, 2]] + values[2] * cc[:, [1, 3]] + values[4]
                        thisContour[:, [1, 3]] = values[1] * cc[:, [0, 2]] + values[3] * cc[:, [1, 3]] + values[5]
                        
                    else:
                        print(f'Unknown transformation: {thisCommand}')
        
        # If the contour needs to be broken up because of M or m path commands,
        # save each piece separately
        contourBreaks = np.append(contourBreaks, thisContour.shape[0] + 1)
        for b in range(len(contourBreaks) - 1):
            vecLD.numContours += 1
            vecLD.contours.append(thisContour[contourBreaks[b] - 1:contourBreaks[b + 1] - 1, :])



def importSVG(svgFilename: str, imsize: np.ndarray = None) -> VecLD:
    """
    Imports a SVG file.
    
    NOTE: This function is experimental. It does not implement all aspects of
    the SVG standard. In particular, it does not translate any text,  
    embedded images, shape fill, or gradients. Some aspects of this function are 
    untested because I couldn't find an SVG file that contain the relevant features. 
    If you find any errors, please email the SVG file that you were trying to 
    load to: dirk.walther@gmail.com, and I will try my best to fix the function. 
    
    Args:
        svgFilename (str): The filename of the SVG file.
        imsize (np.ndarray): The image size (optional). If nothing is provided, the image size will be 
            determined from the SVG file.
        
    Returns:
        result (VecLD): VecLD data structure with the contours from the SVG file.
    """
    
    vecLD = VecLD(
        originalImage = svgFilename,
        imsize = np.empty((1, 2)), # 1 row x 2 col
        lineMethod = __file__, # path of current file
        numContours = 0,
        contours = np.empty(())
    )
    
    tree = xmltree.parse(svgFilename).getroot()
    vecLD = parseChildNodes(tree, vecLD, '')
    
    if imsize is not None:
        vecLD.imsize = imsize
    
    if vecLD.imsize is None:
        maxX = -np.inf
        maxY = -np.inf
        for c in range(vecLD.numContours):
            thisCont = vecLD.contours[c]
            maxX = max(maxX, np.max([thisCont[:,0], thisCont[:,2]]))
            maxY = max(maxY, np.max([thisCont[:,1], thisCont[:,3]]))
        vecLD.imsize = np.ceil([maxX, maxY])
    
    return vecLD

setattr(VecLD, 'importSVG', importSVG)
