import cv2
import dlib
import numpy as np
import warnings
import threading
from playsound import playsound
from scipy.spatial import distance as dist
from tkinter import *
import socket

warnings.simplefilter(action='ignore', category=FutureWarning)
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

#COMMON
def sound_alarm():
        playsound("alarm.wav")

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

# YAWN
def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i].item(1))
    for i in range(61,64):
        top_lip_pts.append(landmarks[i].item(1))
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean)

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i].item(1))
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i].item(1))
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean)

def mouth_open(landmarks,image):
    if landmarks == "error":
        return image, 0
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return lip_distance

#EYES
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def left_eye(landmarks):
    l_eye_points=[]
    for i in range(37,43):
        l_eye_points.append(landmarks[i])
    return l_eye_points

def right_eye(landmarks):
    r_eye_points=[]
    for i in range(43,49):
        r_eye_points.append(landmarks[i])
    return r_eye_points

cap = cv2.VideoCapture(0)

#FOR EYES
eye_count=0
eye_status = False
EYE_AR_THRESH = 4
EYE_AR_CONSEC_FRAMES = 8
COUNTER_EYE=0

#FOR LIPS
yawn_count=0
yawn_status = False
LIP_DIST_LIMIT=30
LIP_DIST_CROSS_CONSEC_FRAMES=3
COUNTER_LIP=0

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    landmarks = get_landmarks(gray) 
    if landmarks=="error":
        continue

    leftEye = left_eye(landmarks)
    rightEye = right_eye(landmarks)
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    prev_eye_status=eye_status
    if ear > EYE_AR_THRESH:
        COUNTER_EYE += 1
        if COUNTER_EYE >= EYE_AR_CONSEC_FRAMES:
            eye_status = True
            if threading.active_count()==2:
                t = threading.Thread(target=sound_alarm)
                t.deamon = True
                t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        COUNTER_EYE = 0
        eye_status=False
    if prev_eye_status == True and eye_status == False:
        eye_count += 1
    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    lip_distance = mouth_open(landmarks,frame)  
    prev_yawn_status = yawn_status
    if lip_distance > LIP_DIST_LIMIT:
        COUNTER_LIP+=1
        if COUNTER_LIP>=LIP_DIST_CROSS_CONSEC_FRAMES:
            yawn_status = True  
            cv2.putText(frame, "Subject is Yawning", (50,450), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
            if threading.active_count()==2:
                t = threading.Thread(target=sound_alarm)
                t.deamon = True
                t.start()      
    else:
        COUNTER_LIP=0
        yawn_status = False          
    if prev_yawn_status == True and yawn_status == False:
        yawn_count += 1
    cv2.imshow('Highway Helper', frame )  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
        
cap.release()
cv2.destroyAllWindows()
stri=f'Yawns: {yawn_count}\nEyes Closed: {eye_count}'
root=Tk()
root.title("Welcome")
l=Label(root,text=stri,font=('Arial',23))
l.grid(row=0,column=0)
root.mainloop()
