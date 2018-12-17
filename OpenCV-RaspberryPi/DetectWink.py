import numpy as np
import cv2
import os, sys
'''
Author: Trey Blankenship 
Date: 5/1/2017 (once I got the bugs ironed out)
This program detects winks. 
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
eye_pair_cascade = cv2.CascadeClassifier("./XML/frontalEyes35x16.xml")
#Mouth (needed for Silence)
mouth_cascade = cv2.CascadeClassifier("./XML/Mouth.xml")

def drawEllipse(frame,rect,rgb):
  #Assume that rect is a tuple in the form of [x,y,w,h]
  x,y,w,h = rect
  width2 = int(w/2)
  height2 = int(h/2)
  center = (x+width2,y+height2)
  cv2.ellipse(frame,center,(width2,height2),0,0,360,rgb,2)

def detectWink(frame,position, roi):
  x,y,w,h = position
  cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
  start_x = x
  start_y = y
  end_x = x + w
  end_y = y + h
  eyes_detected = 0
  eyepairs = eye_pair_cascade.detectMultiScale(roi)
  wink_detected = False
  eyes_detected = False
  if len(eyepairs) >= 1:
    ex,ey,ew,eh = eyepairs[0]#x,y,w,h of eye pair. Ignore width and x, we're only interested in the y values. 
    #cv2.rectangle(frame,(x,ey+y),(x+w,y+ey+eh),(0,255,0),2)
    eye_roi = roi[ey:ey+eh,0:w]
    eyes = eye_2_cascade.detectMultiScale(eye_roi)
    if len(eyes) == 0: 
      eyes = eye_1_cascade.detectMultiScale(eye_roi)
    if len(eyes) == 0: 
      eyes = eye_3_cascade.detectMultiScale(eye_roi)
    if len(eyes) == 0: 
      eyes = left_eye_cascade.detectMultiScale(eye_roi)
    if len(eyes) == 0: 
      eyes = right_eye_cascade.detectMultiScale(eye_roi)
    if len(eyes) == 0: 
      eyes = left_eye_cascade_2.detectMultiScale(eye_roi)
    if len(eyes) == 0: 
      eyes = right_eye_cascade_2.detectMultiScale(eye_roi)
    if len(eyes) > 0: 
      eyes_detected = True    
    for (e_x,e_y,e_w,e_h) in eyes: 
      drawEllipse(frame,(x+e_x,y+ey+e_y,e_w,e_h),(0,0,255))  
    if len(eyes) == 1: 
      wink_detected = True
    
  if (len(eyepairs) == 0) or not eyes_detected:
    #Expand our search to include features not found by our initial search.
    eyes = eye_1_cascade.detectMultiScale(roi)
    if len(eyes) == 0: 
      eyes = eye_2_cascade.detectMultiScale(roi)
    if len(eyes) > 0:
      eyes_dected = True
    for (ex,ey,ew,eh) in eyes: 
      drawEllipse(frame,(x+ex,y+ey,ew,eh),(0,0,255))
    if len(eyes) == 1: 
      wink_detected = True
  return wink_detected
  

def detect(frame):    
    winks_detected = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #Get the grayscale image.     
    #Optimize gray for finding faces. 
    gray = cv2.equalizeHist(gray)
    faces = front_face_cascade_1.detectMultiScale(gray,1.3,3)
    for (x,y,w,h) in faces:
      roi_color = frame[y:y+h,x:x+w]
      roi_gray = gray[y:y+h,x:x+w]
      roi_gray_sub = cv2.GaussianBlur(roi_gray,(5,5),0)
      roi_gray = cv2.addWeighted(roi_gray,1.4,roi_gray_sub,-0.3,0)
      roi_gray = cv2.equalizeHist(roi_gray)
      if detectWink(frame,(x,y,w,h),roi_gray): 
        drawEllipse(frame,(x,y,w,h),(255,0,255))
        winks_detected += 1
    return winks_detected

def runonVideo():
  cap = cv2.VideoCapture(0)
  winks_detected = 0
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
      
    winks_detected += detect(frame)
    cv2.imshow(windowName, frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
      break  
  print("Exiting camera!")
  cap.release()
  cv2.destroyAllWindows()
  return winks_detected
    
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
      
      
