"change name of id -1 and id-2 in for loop"

import cv2,os
import numpy as np
from PIL import Image
from gtts import gTTS
from playsound import playsound
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create() 
language = 'en'
rec.read('recognizer/trainingdata.yml')

font = cv2.FONT_HERSHEY_SIMPLEX

flag1=0
flag2=0
flag3=0
while True:
    ret,img =cam.read()
    

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(conf<50):
            if(id==1):
                id='id-1 person name'
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                if(flag1==0):
                    mytext = 'Hi , Dilkhush , How can i help you'
                    myobj = gTTS(text=mytext, lang=language, slow=False) 
                    myobj.save("welcome.mp3")
                    from playsound import playsound
                    playsound('welcome.mp3')
                    os.remove('welcome.mp3')
                    flag1=flag1+1
                    
            elif(id==2):
                id='id-2 person name'
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                if(flag2==0):
                    mytext = 'Hi ,Anmol , How can i help you'
                    myobj = gTTS(text=mytext, lang=language, slow=False) 
        else:
            id='unkown'
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            if(flag3==0):
                mytext = 'Unknown person detected'
                myobj = gTTS(text=mytext, lang=language, slow=False) 
                myobj.save("welcome.mp3")
                from playsound import playsound
                playsound('welcome.mp3')
                os.remove('welcome.mp3')
                flag3+=1
	    
        print(conf)   
        cv2.putText(img,str(id),(x,y-10), font,1,(0,255,255),2,cv2.LINE_AA)
        cv2.imshow('Face',img)
    if cv2.waitKey(2) & 0xFF==ord('q'):
        break
            
cam.release()
cv2.destroyAllWindows()
