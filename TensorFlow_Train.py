import tensorflow as tf

from numpy import empty
from scipy import rand
import processImages as pi
import csv
import cv2
import numpy as np


def  badlyGetDataFromPix(pixImg):
    blue = [foo[0] for foo in pixImg[0]]
    green = [foo[1] for foo in pixImg[0]]
    red = [foo[2] for foo in pixImg[0]]
    
    pixHsvImg = cv2.cvtColor(pixImg, cv2.COLOR_BGR2HSV)

    hue = [foo[0] for foo in pixHsvImg[0]]
    sat = [foo[1] for foo in pixHsvImg[0]]
    val = [foo[2] for foo in pixHsvImg[0]]

    return([
            np.average(blue), np.std(blue),
            np.average(green), np.std(green),
            np.average(red), np.std(red),
            np.average(hue), np.std(hue),
            np.average(sat), np.std(sat),
            np.average(val), np.std(val),
        ])


if __name__=="__main__":
    dataSet = pi.getFileSet("TensorFlow", tag='_pix.jpg') # Load labeled pixel data
        
    colorSet = []

    pixData = []
    pixLabs = []

    for fileName in dataSet:
        fooColor = fileName.split('/')[-1].split('_')[0] # Isolate color out of name variable
        pixImg = cv2.imread(fileName, 1) # load pixel img
        

        if not fooColor in colorSet:
            colorSet.append(fooColor)
        fooIndex = colorSet.index(fooColor)
        pixLabs.append(fooIndex)

        pixData.append(badlyGetDataFromPix(pixImg))
    
    print(pixLabs)

    pixClassLabs = [[0]*ii + [1] + [0]*(max(pixLabs)-ii) for ii in pixLabs]

    print(pixClassLabs)
    
    # first neural network with keras tutorial
    from numpy import loadtxt
    from keras.models import Sequential
    from keras.layers import Dense
    # load the dataset

    # define the keras model
    model = Sequential()
    model.add(Dense(15, input_dim=12, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(len(colorSet), activation='softmax'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(pixData, pixClassLabs, epochs=150, batch_size=10)
    # evaluate the keras model
    _, accuracy = model.evaluate(pixData, pixClassLabs)
    print('Accuracy: %.2f' % (accuracy*100))

    predictions = model.predict(pixData)
    for ii in range(len(predictions)):
        print(colorSet[pixLabs[ii]] + '   ' + str(np.where(predictions[ii] == (max(predictions[ii]))) ) )

    # serialize model to JSON
    model_json = model.to_json()
    with open("TensorFlow/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("TensorFlow/model.h5")
    print("Saved model to disk")

    with open('TensorFlow/colors.txt', 'w') as color_file:
        for foo in colorSet:
            color_file.write(foo)
            color_file.write(',')


    # import random as r
    # outFile = open("trashData.csv", 'w')

    # for ii in range(5000):
    #     aaa = int(rand()>0.5)
    #     bbb = rand()*100

    #     ccc = rand()*50
    #     if aaa > 0: ccc += rand()*50-20
        
    #     ddd = rand()*100
    #     if aaa == 0: ddd -= rand()*50 + 20
    #     else: ddd += rand()*50

    #     outStr = str(int(bbb)) + ', ' + str(int(ccc)) + ', ' + str(int(ddd)) + ', ' + str(aaa) + '\n'
    #     outFile.write(outStr)
