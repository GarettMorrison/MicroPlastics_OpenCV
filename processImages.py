# importing the module
from lib2to3.pgen2.token import NEWLINE
from re import L
from telnetlib import XASCII
from matplotlib.cbook import maxdict
from pandas import concat
import colorsys
import cv2
import numpy as np
import math as m
import os



def cropToScope(img):

    # print(img)
    yMin = int(img.shape[1] * 0.38)
    yMax = int(img.shape[1] * 0.65)
    
    xCrop = int(img.shape[0] * 0.4)
    # xMax = int(img.shape[0] * 0.9)

    return(img[yMin:yMax, xCrop:-1*xCrop])


def value_filter(img, valueMin):
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
    kernel = np.ones((10,10),np.uint8)
    imgOut = cv2.morphologyEx(img, cv2.MORPH_ELLIPSE, kernel)
    return(imgOut)


def getDist(pt1, pt2):
    return(m.sqrt( (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 ))


def farthestPoint(basePt, pts):
    maxDist = 0
    maxPt = (0,0)
    for fooPt in pts:
        fooDist = getDist(fooPt, basePt)
        if fooDist > maxDist:
            maxDist = fooDist
            maxPt = fooPt
    return(maxPt)


def getMaxWidth(img):
    pts = []
    for yy in np.arange(img.shape[0]):
        for xx in np.arange(img.shape[1]):
            if(img[yy][xx] != 0):
                pts.append((xx, yy))

    xAvg = sum([ii[0] for ii in pts]) / len(pts)
    yAvg = sum([ii[1] for ii in pts]) / len(pts)
    midPt = (xAvg, yAvg)

    pt1 = farthestPoint(midPt, pts)
    pt2 = farthestPoint(pt1, pts)
    maxDist = getDist(pt1, pt2)

    outImg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    outImg = cv2.line(outImg, pt1, pt2, (0,0,255), 2)
    outImg = cv2.circle(outImg, (int(midPt[0]), int(midPt[1])), 5, (255,0,0), 2)

    return(outImg, maxDist)


def scaleFrac(img, scaleFrac):
    dims = [int(img.shape[1]*scaleFrac), int(img.shape[0]*scaleFrac)]
    imScaled = cv2.resize(img, dims, interpolation = cv2.INTER_AREA)
    return(imScaled)


def adjacentImages(imgArr):
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
    # getFileSet('FolderName', tag='jpg')
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

    #range of images to check
    
    dataSet = getFileSet("TestData", tag='.jpg')
    print(dataSet)

    for fileName in dataSet:   
        imgRaw = cv2.imread(fileName, 1)
        name = fileName.split('/')[-1]
        
        # img = cv2.imread('Data/MISPFeb2501.jpg', 1)
    
        img = cropToScope(imgRaw)
        # imgFilt = value_filter(img, 120)
        imgEdge = edge_filter(img)
        # imgEdgeFilt = value_filter(imgEdge, 80)
        imgEdgeFilt = value_filter(imgEdge, 70)

        imgCanny = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
        imgEdgeCanny = cv2.cvtColor(cv2.Canny(imgEdge, 100, 200), cv2.COLOR_GRAY2BGR)
        imgEdgeFiltCanny_gray = cv2.Canny(imgEdgeFilt, 100, 200)

        imgEdgeFiltCanny, maxWidth = getMaxWidth(imgEdgeFiltCanny_gray)
        particleSize = maxWidth / 575.6808510638298

        imCombined = adjacentImages(
            [[img, imgEdge],
            [imgEdgeFilt, imgEdgeFiltCanny]])
            
        # imScaled = scaleFrac(imCombined, 1)
        imOut = imCombined
        
        imOut = cv2.putText(imOut, 
            str('px:'+ str(round(maxWidth, 3))),
            (int(imOut.shape[1]/2), int(imOut.shape[0]/2)+32), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (255,0,255), 2, cv2.LINE_AA)

        imOut = cv2.putText(imOut, 
            str('cm:'+ str(round(particleSize, 3))),
            (int(imOut.shape[1]/2), int(imOut.shape[0]/2)+64), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (255,0,255), 2, cv2.LINE_AA)
        
        cv2.imwrite('OutputPictures/filt_' + str(name), imOut)

        print(str(name) + ' : ' + str(particleSize))
        
        fileOut = open('outData.csv', 'a')
        fileOut.write(fileName + ', ' + str(name) + ', '  + str(particleSize) + ', ' + str(maxWidth) + '\n')
        fileOut.close()

        # cv2.imshow('image', imOut)

        # # wait for a key to be pressed to exit
        # cv2.waitKey(0)
    
        # # close the window
        # cv2.destroyAllWindows()