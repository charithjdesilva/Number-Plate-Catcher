import cv2

import cv2
import numpy as np
from image_stacking import stackImages

# add cascade from openCV library to detect faces
nbPlateCascade = cv2.CascadeClassifier("../chapter 1/resources/haarcascades/haarcascade_russian_plate_number.xml")

imgWidth = 640
imgHeight = 480
minArea = 500    # minimum detection area
color = (255,255,0)
counter = 0


# videoCap = cv2.VideoCapture(2)
videoCap = cv2.VideoCapture("https://192.168.8.100:4343/video") # chnage accoring to the ip and port of the cam
videoCap.set(3, imgWidth)
videoCap.set(4, imgHeight)
videoCap.set(10, 150)
# videoCap.set(cv2.CAP_PROP_BRIGHTNESS , 150)   


while True:
    success, img = videoCap.read()
    img = cv2.resize(img,(imgWidth, imgHeight)) # resize the image
    imgCrop = img[10:480, 0:640]
    imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)

    # find the number plates using the number plate cascade
    numberPlates = nbPlateCascade.detectMultiScale(imgGray, 1.1, 4)

    # count = 0
    # create a bounding box around the detected number plates
    for (x, y, width, height) in numberPlates:
        # detect number plates greater than this area
        area = width*height
        if area > minArea:
            cv2.rectangle(imgCrop, (x,y), (x+width, y+height), (255, 255, 0), 2)
            cv2.putText(imgCrop, "Number Plate", (x, y-5),   #put text on the image above the detected bounding box
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            imgReigionOfInterest = img[y:y+height, x:x+width]      # extract number plate area
            cv2.imshow("Region of Interest",imgReigionOfInterest)
            # cv2.imshow("Region of Interest "+str(counter),imgReigionOfInterest)
            # count += 1

    cv2.imshow("Result", imgCrop)

    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('s'):   # save image on 's' key press
        # save the number plate and get a feedback that it was saved
        cv2.imwrite("outputs/scanned/Number Plate_"+str(counter)+".jpg",imgReigionOfInterest)
        cv2.rectangle(imgCrop,(0,200),(640,300),(153,102,102),cv2.FILLED)   # feedback on a rectangle
        cv2.putText(imgCrop, "Scan Saved", (150,265), cv2.FONT_HERSHEY_COMPLEX,
                    2, (0,0,0),2)
        cv2.imshow("Result", imgCrop)
        cv2.waitKey(500)    #wait 500 mili seconds after saving
        counter += 1
    elif pressedKey == ord('q'):
        break