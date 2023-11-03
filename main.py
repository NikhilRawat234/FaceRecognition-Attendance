import cv2
import numpy as np
import face_recognition
import logging as log
import os
from datetime import datetime



path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print('Loading...')

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print('Wait a moment.....')



 
def findEncodings(images):
    encodeList = []
    for img in images:
        if(img is not None):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
    return encodeList


 
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0]) 
        if name not in nameList:
            now = datetime.now()
            dtString1 = now.strftime('%Y-%m-%d') 
            dtString = now.strftime('%H:%M:%S')
            result="Present"
            f.writelines(f'{name}    ,{dtString1},{dtString} ,{result}\n')



            
            
hasOpened = False
            
 
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
 
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()

    

    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
 
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            if success and not hasOpened:
                log.warning('Camera Opened Successfully')
                hasOpened = True
                markAttendance(name)
        
       
 
        cv2.imshow('Webcam',img)
        key = cv2.waitKey(1)
        if cv2.waitKey(1) & key == ord('q'):
            print("Turning off camera.")
            print("Camera off.")
            print("Program ended.") 
            cap.release()
            cv2.destroyAllWindows()
            break
        




        
        if key == ord('q'):
            print("Turning off camera.")
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break