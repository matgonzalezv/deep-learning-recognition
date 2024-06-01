import cv2 as cv
import os
import imutils

model = 'MarkPhotos'
path1 = 'C:/Users/USER/Desktop/facialrecognition1'
fullpath = path1 + '/' + model
if not os.path.exists(fullpath):
    os.makedirs(fullpath)

camera = cv.VideoCapture('mark.mp4')
noise = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
id = 0
while True:
    response, capture = camera.read()
    if response == False:
        break
    capture = imutils.resize(capture, width=640)

    gray = cv.cvtColor(capture, cv.COLOR_BGR2GRAY)
    capture_id = capture.copy()

    # x left side, y from top to bottom, e1 and e2 corner parts
    face = noise.detectMultiScale(gray, 1.3, 5)

    for (x, y, e1, e2) in face:
        cv.rectangle(capture, (x, y), (x+e1, y+e2), (0, 255, 0), 2)
        face_captured = capture_id[y:y+e2, x:x+e1]
        face_captured = cv.resize(face_captured, (160, 160), interpolation=cv.INTER_CUBIC)
        cv.imwrite(fullpath+'/image_{}.jpg'.format(id), face_captured)
        id = id+1

    cv.imshow("Face Result", capture)

    if id == 351:
        break
camera.release()
cv.destroyAllWindows()
