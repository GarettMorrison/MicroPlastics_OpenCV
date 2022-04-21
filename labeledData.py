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

import processImages as pi


# driver function
if __name__=="__main__":
    # Get set of image file names     
    # dataSet = getFileSet('TestData', tag='.jpg')
    dataSet = pi.getFileSet("DataIn", tag='.jpg')

    # Get labeled data and only pull pics out of this ID set
    # Must have collumns labeled 'ID' and 'Color'
    labelData = pandas.read_csv('BigData/MISPFeb_labeled.csv')
    
    print('Starting loop')

    ii = 0 # Start at first file    
    while True:
        if ii >= len(dataSet): break # end if at end of file list

        # Remove if not in labeled data
        isIncluded = False
        for fooID in labelData['ID']:
            if dataSet[ii].find(fooID) > 0:
                isIncluded = True
                break

        if isIncluded:
            ii += 1
        else:   # if tag not in name, remove
            dataSet.pop(ii)

    # Get dictionary of how many of each color has been processed
    colorSavedCounts = {}
    for fooID in set(labelData['Color']):
        colorSavedCounts[fooID] = 0


    for fileName in dataSet:  
        name = fileName.split('/')[-1][:-4] # get just filename
        labelIndex = labelData[labelData['ID'] == name]
        color = labelIndex['Color'].iloc[0]
        
        imgRaw = cv2.imread(fileName, 1) # load image
        img = pi.cropToScope(imgRaw)   # Crop image to just particle
        imgEdge = pi.edge_filter(img)      # Run edge filter to elimate noisy background
        imgEdgeFilt = pi.value_filter(imgEdge, 70)     # Value filter to just get image

        # Get particle pixels and save as image
        containedPix = pi.pixFromSelection(img, imgEdgeFilt)
        cv2.imwrite('LabeledPixelData/' + str(color) + '_' + str(colorSavedCounts[color]) + '_pix.jpg', np.array([containedPix]) ) # Save process image
        print('Saved ' + str(color) + '_' + str(colorSavedCounts[color]) + '_pix.jpg')
        colorSavedCounts[color] += 1

        # # Image display stuff
        # imgBack = pi.blackBackroundFromSelection(img, imgEdgeFilt)
        # imgOut = pi.adjacentImages(
        #     [[img, imgEdge],
        #     [imgEdgeFilt, imgBack]])

        # # imgOut = imgEdgeFilt
        # imgOut = cv2.putText(imgOut, 
        #     str('px:'+ color),
        #     (int(imgOut.shape[1]/2), int(imgOut.shape[0]/2) +33), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 
        #     1, (255,0,255), 2, cv2.LINE_AA)
        
        # imgOut = pi.scaleFrac(imgOut, 0.75)


        
        # Display each image on run
        # cv2.imshow('image', imgOut)
        # keypress = cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # if keypress == 113:
        #     print('Exiting display loop')
        #     break
        