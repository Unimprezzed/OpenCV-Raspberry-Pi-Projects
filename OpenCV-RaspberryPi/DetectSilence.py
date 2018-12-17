import numpy as np
import cv2
import os, sys
'''
Author: Trey Blankenship 
Date: 5/1/2017 (once I got the bugs ironed out)
This program detects silences 
'''
#Front face. 
default_face_cascade = cv2.CascadeClassifier("./XML/haarcascade_frontalface_default.xml")
front_face_cascade_1= cv2.CascadeClassifier("./XML/haarcascade_frontalface_alt2.xml")
front_face_cascade_2 = cv2.CascadeClassifier("./XML/haarcascade_frontalface_alt.xml")
face_tree_cascade = cv2.CascadeClassifier("./XML/haarcascade_frontalface_alt_tree.xml")
#Face on profile.
profile_face_cascade = cv2.CascadeClassifier("./XML/haarcascade_profileface.xml")
#Eyes (we don't care which)

eye_1_cascade = cv2.CascadeClassifier("./XML/haarcascade_eye_tree_eyeglasses.xml")
eye_2_cascade = cv2.CascadeClassifier("./XML/haarcascade_eye.xml")
eye_3_cascade = cv2.CascadeClassifier("./XML/haarcascade eye.xml")
left_eye_cascade = cv2.CascadeClassifier("./XML/haarcascade_righteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("./XML/haarcascade_lefteye_2splits.xml")
left_eye_cascade_2 = cv2.CascadeClassifier("./XML/ojoI.xml")
right_eye_cascade_2 = cv2.CascadeClassifier("./XML/ojoD.xml")

#Eye pairs
eye_pair_cascade_1 = cv2.CascadeClassifier("./XML/frontalEyes35x16.xml")
eye_pair_cascade_2 = cv2.CascadeClassifier("./XML/parojos.xml")
eye_pair_cascade_3 = cv2.CascadeClassifier("./XML/parojosG.xml")
#Nose 
nose_cascade = cv2.CascadeClassifier("./XML/Nariz.xml")
#Mouth (needed for Silence)
mouth_cascade = cv2.CascadeClassifier("./XML/Mouth.xml")

def drawEllipse(frame,rect,rgb):
  #Assume that rect is a tuple in the form of [x,y,w,h]
  x,y,w,h = rect
  width2 = int(w/2)
  height2 = int(h/2)
  center = (x+width2,y+height2)
  cv2.ellipse(frame,center,(width2,height2),0,0,360,rgb,2)

def detectSilence(frame,position, roi):
  x,y,w,h = position
  cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
  start_x = x
  start_y = y
  end_x = x+w
  end_y = y+h
  #Look for eye pairs.
  h2 = int(h*0.6) 
  roi_color = frame[y:y+h,x:x+w]  
  roi_gray = roi[h2:h,0:w]
  roi_gray = cv2.equalizeHist(roi_gray)
  '''
    #Look for eye pairs.
  eyes = eye_pair_cascade_1.detectMultiScale(roi)
  if len(eyes) == 0:
    eyes = eye_pair_cascade_2.detectMultiScale(roi)
  if len(eyes) == 0: 
    eyes = eye_pair_cascade_3.detectMultiScale(roi)  
  if len(eyes) == 0: 
    #If that does not work, look for eyes instead. We need only one per face to constrain our search
    eyes = eye_1_cascade.detectMultiScale(roi)
  if len(eyes) == 0: 
    eyes = eye_2_cascade.detectMultiScale(roi)
  if len(eyes) == 0: 
    eyes = eye_3_cascade.detectMultiScale(roi)
 
  for (ex,ey,ew,eh) in eyes: 
    cv2.rectangle(frame,(x,ey+y),(x+w,y+ey+eh),(0,255,0),2)  
    '''
  mouths = mouth_cascade.detectMultiScale(roi_gray,1.305,5)
  for(mx,my,mw,mh) in mouths: 
    drawEllipse(frame, (x+mx,y+my+h2,mw,mh), (255,255,0))  
  return len(mouths) == 0

def detect(frame):    
    silences_detected = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #Get the grayscale image.     
    #Optimize gray for finding faces for this problem
    g_blur = cv2.bilateralFilter(gray,9,75,75)
    gray = cv2.addWeighted(gray,1.4,g_blur,-0.3,0)
    gray = cv2.equalizeHist(gray)
    
    faces = front_face_cascade_1.detectMultiScale(gray,1.1,3)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    for (x,y,w,h) in faces: 
      roi_color = frame[y:y+h,x:x+w]
      roi_gray = gray[y:y+h,x:x+w]
      roi_gray_sub = cv2.GaussianBlur(roi_gray,(5,5),0)
      roi_gray = cv2.addWeighted(roi_gray,1.4,roi_gray_sub,-0.3,0)
      roi_gray = cv2.equalizeHist(roi_gray)
      if detectSilence(frame,(x,y,w,h),roi_gray):
        drawEllipse(frame,(x,y,w,h),(255,0,255))
        silences_detected += 1
    return silences_detected

def runonVideo():
  cap = cv2.VideoCapture(0)
  silences_detected = 0
  if(not cap.isOpened()):
    print("Could not open default video camera!")
    exit(1)
  
  windowName = 'Live Video Feed'
  
  cv2.namedWindow(windowName,cv2.WINDOW_AUTOSIZE)
  
  while(True):
    ret, frame = cap.read()
    if(not ret):
      print("Could not read frame!")
      break
      
    silences_detected += detect(frame)
    cv2.imshow(windowName, frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
      break  
  print("Exiting camera!")
  cap.release()
  cv2.destroyAllWindows()
  return silences_detected
    
def runonFolder(folder):
    detections = 0
    windowName = ""
    imagefiles=[os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f))]
    imagefiles.sort() #In the order they're displayed by the operating system.
    for f in imagefiles:
        img = cv2.imread(f)
        if(len(img) != 0):
            d = detect(img)
            detections += d
            #print("windowName = '" + windowName + "'")
            w_len = len(windowName)
            if(w_len != 0):
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return detections

if __name__ == "__main__":
    if(len(sys.argv) == 1):
      m_detections = runonVideo()
      print(str(m_detections) + " winks detected in live feed.")
    elif(len(sys.argv) == 2):
      m_detections = runonFolder(sys.argv[1])
      print("Total of " + str(m_detections) + " detected in folder." )
      
      
