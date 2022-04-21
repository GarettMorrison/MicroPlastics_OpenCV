# importing the module
from lib2to3.pgen2.token import NEWLINE
from re import L
from telnetlib import XASCII
from matplotlib.cbook import maxdict
import pandas
from pandas import concat
import colorsys
import cv2
import numpy as np
import math as m
import os
import csv



def cropToScope(img, xCrop=(0,1), yCrop = (0,1)):
    # Crop image to specific ranges to just get particle    
    leftCrop = int(img.shape[0] * xCrop[0])
    rightCrop = int(img.shape[0] * xCrop[1])

    topCrop = int(img.shape[1] * yCrop[0])
    botCrop = int(img.shape[1] * yCrop[1])

    return(img[topCrop:-1*botCrop, leftCrop:-1*rightCrop])

def value_filter(img, valueMin):
    # Filter out all pixels with values above valueMin
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for yy in np.arange(img.shape[0]):
        for xx in np.arange(img.shape[1]):
            pixel = img[yy][xx]

            if pixel[2] > valueMin:
                img[yy][xx] = [0, 255, 255]
            else:
                img[yy][xx] = [0, 0, 0]
    
    return(cv2.cvtColor(img, cv2.COLOR_HSV2BGR))


def edge_filter(img):
    # Filter out noise, get solid particle outline
    kernelDim = 13
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelDim, kernelDim))
    # kernel = np.ones((kernelDim, kernelDim),np.uint8)
    imgOut = cv2.morphologyEx(img, cv2.MORPH_ELLIPSE, kernel)
    return(imgOut)


def pixFromSelection(imgBase, imgSelect):
    # Get a list of pixels from imgBase where imgSelect has color
    pixels = []

    for yy in np.arange(img.shape[0]):
        for xx in np.arange(img.shape[1]):
            if((imgSelect[yy][xx]).any() != 0):
                pixels.append(imgBase[yy][xx])
                # print(imgBase[yy][xx])

    return(np.array(pixels))


def blackBackroundFromSelection(imgBase, imgSelect):
    # Get a list of pixels from imgBase where imgSelect has color
    img = np.array(imgBase) # Get empty array like the base image

    for yy in np.arange(imgBase.shape[0]):
        for xx in np.arange(imgBase.shape[1]):
            if not (imgSelect[yy][xx]).any() != 0:
                img[yy][xx] = [0,0,0]

    return(img)


def getDist(pt1, pt2): # Simple distance function
    return(m.sqrt( (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2) )


def farthestPoint(basePt, pts): 
    # Get point that is farthest from the base
    maxDist = 0
    maxPt = (0,0)
    for fooPt in pts:
        fooDist = getDist(fooPt, basePt)
        if fooDist > maxDist:
            maxDist = fooDist
            maxPt = fooPt
    return(maxPt)


def getMaxWidth(img):
    # Get maximum width of outline image
    
    outImg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    pts = []
    for yy in np.arange(img.shape[0]):
        for xx in np.arange(img.shape[1]):
            if(img[yy][xx] != 0):
                pts.append((xx, yy))
    
    if len(pts) <= 0: return(outImg, 0)

    xAvg = sum([ii[0] for ii in pts]) / len(pts)
    yAvg = sum([ii[1] for ii in pts]) / len(pts)
    midPt = (xAvg, yAvg)

    pt1 = farthestPoint(midPt, pts)
    pt2 = farthestPoint(pt1, pts)
    maxDist = getDist(pt1, pt2)


    outImg = cv2.line(outImg, pt1, pt2, (0,0,255), 2)
    outImg = cv2.circle(outImg, (int(midPt[0]), int(midPt[1])), 5, (255,0,0), 2)

    return(outImg, maxDist)


def scaleFrac(img, scaleFrac): # easily scale image by constant value
    dims = [int(img.shape[1]*scaleFrac), int(img.shape[0]*scaleFrac)]
    imScaled = cv2.resize(img, dims, interpolation = cv2.INTER_AREA)
    return(imScaled)


def adjacentImages(imgArr): # Place images into grid for easy comparison
    blankImg = np.zeros((imgArr[0][0].shape[0], imgArr[0][0].shape[1], 3))

    rowMax = max([len(ii) for ii in imgArr])
    # rowMax = max([len(ii) for ii in imgArr])
    
    for fooRow in imgArr:
        while (len(fooRow) < rowMax):
            fooRow.append(blankImg)

    imgRows = []
    for fooRow in imgArr:
        concatRow = np.concatenate(fooRow, axis=1)
        # if(len(fooRow) < rowMax):
        #     concatRow = np.concatenate([concatRow] + [blankImg]*(rowMax - len(fooRow)), axis=1)
        
        imgRows.append(concatRow)
    

    return(np.concatenate(imgRows, axis=0))


def getFileSet(folderName, recursive=False, tag = ''):
    # get array of file names as strings, can specify a tag
    # ex: "getFileSet('FolderName', tag='jpg')
    outFiles = []

    if not recursive:
        outFiles = os.listdir(folderName)

    ii = 0 # Start at first file    
    while tag != '':
        if ii >= len(outFiles): break # end if at end of file list
        
        if outFiles[ii].find(tag) > 0: # if tag in name, continue
            ii += 1
        else: # if tag not in name, remove
            outFiles.pop(ii)
    
    for ii in range(len(outFiles)):
        outFiles[ii] = folderName + '/' + outFiles[ii]
    
    return(outFiles)



# driver function
if __name__=="__main__":
    # Reset output file
    fileOut = open('outData.csv', 'w')
    fileOut.write('File Name, File Index,  Size (cm), Size (px)\n')
    fileOut.close()

    # Get set of image file names     
    dataSet = getFileSet('TestData', tag='.jpg')
    # dataSet = getFileSet("DataIn", tag='.jpg')
    
    print('Starting process')

    for fileName in dataSet:  
        imgRaw = cv2.imread(fileName, 1) # load image
        name = fileName.split('/')[-1] # get just filename

        # img = imgRaw
        img = cropToScope(imgRaw, (0.4, 0.4), (0.4, 0.15))   # Crop image to just particle, numbers are calibrated constants for example data
        imgEdge = edge_filter(img)      # Run edge filter to elimate noisy background
        imgEdgeFilt = value_filter(imgEdge, 70)     # Value filter to just get image
        imgEdgeFiltCanny_gray = cv2.Canny(imgEdgeFilt, 100, 200)       # Outline particle

        imgEdgeFiltCanny, maxWidth = getMaxWidth(imgEdgeFiltCanny_gray)     # Get maximum particle width
        particleSize = maxWidth / 575.6808510638298     # Convert to mm poorly 

        # Combine process images
        imCombined = adjacentImages(
            [[img, imgEdge],
            [imgEdgeFilt, imgEdgeFiltCanny]])
            
        imOut = imCombined
        
        # Display max width in pixels
        imOut = cv2.putText(imOut, 
            str('px:'+ str(round(maxWidth, 3))),
            (int(imOut.shape[1]/2), int(imOut.shape[0]/2)+32), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (255,0,255), 2, cv2.LINE_AA)

        # Display max width in mm
        imOut = cv2.putText(imOut, 
            str('cm:'+ str(round(particleSize, 3))),
            (int(imOut.shape[1]/2), int(imOut.shape[0]/2)+64), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (255,0,255), 2, cv2.LINE_AA)
        
        
        cv2.imwrite('OutputPictures/filt_' + str(name), imOut) # Save process image

        print(str(name) + ' : ' + str(particleSize))
        
        # Save data to output csv
        fileOut = open('outData.csv', 'a')
        fileOut.write(fileName + ', ' + str(name) + ', '  + str(particleSize) + ', ' + str(maxWidth) + '\n')
        fileOut.close()

        # Display each image on run
        cv2.imshow('image', imOut)
        keypress = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if keypress == 113: # if key hit was q, exit without displaying more
            break