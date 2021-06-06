import cv2
import numpy as np
import math
import time

def phase1(robot):
    x1=robot.ee_pos()[0]
    x2=robot.block_pos(robot.boxId[0])[0]
    dx = 0.01
    direction = 1 if x2-x1>0 else -1
    while abs(x1-x2)>0.005:
        pos=robot.ee_pos()
        pos[0]+=dx*direction
        x1=pos[0]
        robot.move_pos(pos)
    print('move x')

def phase4(robot):
    for _ in range(20):
        ee_pos=robot.ee_pos()
        ee_pos[1]+=0.01
        robot.move_pos(ee_pos)
    print("move back")

def phase3(robot):
    ee_pos=robot.ee_pos()
    goal_pos=robot.block_pos(robot.boxId[0])
    step=0
    k=1
    while distance(ee_pos,goal_pos)>0.01:
        ee_pos=robot.ee_pos()
        step+=1
        if step>10:
            k=10
        if step>50:
            break
        dx,dy=delta(goal_pos[0],ee_pos[0],goal_pos[1],ee_pos[1])
        dir=-1 if (ee_pos[0]-goal_pos[0])>0 else 1
        if (ee_pos[1]-goal_pos[1])<0:
            if abs(ee_pos[0]-goal_pos[0])<0.005:
                break
            ee_pos[0]+=dx*dir/k
        else:
            ee_pos[0]+=dx*dir/k
            ee_pos[1]-=dy
        robot.move_pos(ee_pos)    
    print('move goal')
    return True

def phase2(robot,ID,left):
    ee_pos=robot.ee_pos()
    ob_pos=robot.block_pos(ID)
    goal_pos=[ob_pos[0]+left*0.09,ob_pos[1]+0.04,ob_pos[2]]
    step=0
    while distance(ee_pos,goal_pos)>0.01:
        step+=1
        if step>50:
            break
        ee_pos=robot.ee_pos()
        ob_pos=robot.block_pos(ID)
        #time.sleep(0.1)       
        goal_pos=[ob_pos[0]+left*0.09,ob_pos[1]+0.04,ob_pos[2]]
        dx,dy=delta(goal_pos[0],ee_pos[0],goal_pos[1],ee_pos[1])
        if abs(ee_pos[0]-ob_pos[0])<0.08 and abs(ee_pos[1]-ob_pos[1])<0.02:
            print('stuck')
            for _ in range(5):
                ee_pos=robot.ee_pos()
                ee_pos[1]+=0.01
                robot.move_pos(ee_pos)
            for _ in range(5):
                ee_pos=robot.ee_pos()
                dir= 1 if dx>0 else -1
                ee_pos[0]-= 0.01*dir
                robot.move_pos(ee_pos)
                
        elif (ee_pos[0]-goal_pos[0])>0 and (ee_pos[1]-goal_pos[1])<0:
            break
        elif (ee_pos[0]-goal_pos[0])>0:
            ee_pos[1]-=dy
        elif (ee_pos[1]-goal_pos[1])<-0.02:
            ee_pos[0]+=dx
        else:
            ee_pos[0]+=dx
            ee_pos[1]-=dy
        robot.move_pos(ee_pos)
    print('reach obstacle')
    for _ in range(10):
        time.sleep(0.1)
        ee_pos=robot.ee_pos()
        ee_pos[0]-=0.01*left
        robot.move_pos(ee_pos)
    print('push obstacle')        

def distance(pos1,pos2):
    x1=pos1[0]
    x2=pos2[0]
    y1=pos1[1]
    y2=pos2[1]
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def delta(x1,x2,y1,y2):
    dis=math.sqrt((x1-x2)**2+(y1-y2)**2)
    dx=abs(x1-x2)/dis*0.01
    dy=abs(y1-y2)/dis*0.01
    dx=max(dx,0.001)
    dy=max(dy,0.001)
    return dx,dy