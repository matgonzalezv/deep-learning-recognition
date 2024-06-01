import cv2 as cv
import os
import imutils


dataPath = 'C:/Users/USER/Desktop/facialrecognition1/Data'
dataList = os.listdir(dataPath)
eigenFaceRecognizerTraining = cv.face.EigenFaceRecognizer_create()
eigenFaceRecognizerTraining.read('EigenFaceRecognizerTraining.xml')
noise = cv.CascadeClassifier(
    'C:/Users/USER/Desktop/facialrecognition1/entrenamientos opencv ruidos/opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml')
camera = cv.VideoCapture('mark.mp4')
while True:
    response, capture = camera.read()
    if response == False:
        break
    capture = imutils.resize(capture, width=640)
    gray = cv.cvtColor(capture, cv.COLOR_BGR2GRAY)
    capture_id = gray.copy()
    face = noise.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face:
        face_captured = capture_id[y:y+h, x:x+w]
        face_captured = cv.resize(
            face_captured, (160, 160), interpolation=cv.INTER_CUBIC)
        result = eigenFaceRecognizerTraining.predict(face_captured)
        cv.putText(capture, '{}'.format(result), (x, y-5),
                   1, 1.3, (0, 255, 0), 1, cv.LINE_AA)
        if result[1] < 9000:
            cv.putText(capture, '{}'.format(
                dataList[result[0]]), (x, y-20), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(capture, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            cv.putText(capture, "Not found", (x, y-20),
                       2, 0.7, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(capture, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv.imshow("Results", capture)
    if cv.waitKey(1) == ord('s'):
        break
camera.release()
cv.destroyAllWindows()
