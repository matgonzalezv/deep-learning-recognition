import cv2 as cv
import cv2
import os
import numpy as np
from time import time

dataPath = 'C:/Users/USER/Desktop/facialrecognition1/Data'
dataList = os.listdir(dataPath)
# print('data', dataList)
ids = []
facesData = []
id = 0
initialTime = time()
for folder in dataList:
    fullPath = dataPath + '/' + folder
    print('Starting reading...')
    for file in os.listdir(fullPath):

        print('Images: ', folder + '/' + file)

        ids.append(id)
        facesData.append(cv.imread(fullPath + '/' + file, 0))

    id = id + 1
    finalReadTime = time()
    totalReadTime = finalReadTime - initialTime
    print('Total read time: ', totalReadTime)

eigenFaceRecognizerTraining = cv.face.EigenFaceRecognizer_create()
print('Starting training...please wait')
eigenFaceRecognizerTraining.train(facesData, np.array(ids))
finalTrainingTime = time()
totalTrainingTime = finalTrainingTime - totalReadTime
eigenFaceRecognizerTraining.write('EigenFaceRecognizerTraining.xml')
print('Training completed')
