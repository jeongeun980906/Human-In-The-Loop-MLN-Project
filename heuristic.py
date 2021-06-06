#from toy import robot_HIL
from robot import robot_HIL
import cv2
import numpy as np
import time
import copy
from heuristic_planner import *
cx,cy=None,None

def callback(event,x,y,flags,param):
    global cx,cy
    if event == cv2.EVENT_LBUTTONDOWN:
        cx=int(x)
        cy=int(y)
        #print(cx,cy)

def color(img):
    cv2.circle(img, (cx,cy), 3,(0,0,255),-1)
    return img

robot=robot_HIL(N=6)
cv2.namedWindow('image')
cv2.setMouseCallback('image',callback)
scale=2
for _ in range(100):
    pos=robot.ee_pos()
    pos[1]-=0.05
    robot.move_pos(pos)
    time.sleep(1)
    phase=1
    print(robot.boxId)
    while True:
        pos=robot.ee_pos()
        img = robot.get_img()
        img = cv2.resize(img,None,fx=scale,fy=scale)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('image',img)
        if cx is not None:
            img=color(img)
        ID,left= robot.get_path(0.05)
        print(robot.stuck())
        if robot.stuck():
            phase4(robot)
            phase=1
        if phase==1:
            #print('1')
            phase1(robot)
            ID,left= robot.get_path(0.05)
            print(ID,left)
            if left is not None:
                phase=2
            else:
                phase=3
        elif phase==3:
            s=phase3(robot)
            if s:
                print("Done")
                break
            else:
                phase=1
        elif phase==2 and left is not None:
            phase2(robot,ID,left)
            ID,left= robot.get_path(0.05)
            print(ID,left)
            if left==None:
                phase=1
        if robot.goal==True:
            break
        k = cv2.waitKey(1) & 0xFF
        if k==ord('q'):
            break
    time.sleep(1)
    robot.reset()  
    
        #print(robot.goal())    