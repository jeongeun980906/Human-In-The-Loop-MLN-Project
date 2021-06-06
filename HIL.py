#from toy import robot_HIL
from robot import robot_HIL
import cv2
import numpy as np
import time
import copy
from heuristic_planner import *
from MLN.train import Agent
import matplotlib.pyplot as plt
import torch

def draw(img,marker):
    cv2.putText(img,'w:move x, e:move y, r: back, num', (0,10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,0,0),1, cv2.LINE_AA)
    for i in range(len(marker)):
        #print(marker[i])
        cv2.putText(img,str(i+2),marker[i],cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5,(255,255,0),1, cv2.LINE_AA)
    return img

N=5+1+1+1+1
robot=robot_HIL(N-3)
cv2.namedWindow('image')
scale=2
agent =Agent(action_size=N,state_size=[N-2,2],lr=1e-4,thres=1.5,ratio2=1)
global_step=0
UE=[]
HUMAN=[]
flag=0
np.random.seed(seed=0)
torch.manual_seed(seed=100)
while flag<4:
    local_step=0
    ue_avg=0
    human=0
    pos=robot.ee_pos()
    pos[1]-=0.05
    robot.move_pos(pos)
    time.sleep(1)
    phase=1
    phase1(robot)
    while True:
        pos=robot.ee_pos()
        img = robot.get_img()
        marker=robot.marker()
        img = cv2.resize(img,None,fx=scale,fy=scale)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = draw(img,marker)
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k==ord('q'):
            break
        s=robot.state()
        policy,thres,ue=agent.play(s)
        if global_step<2 or thres:
            print("Need Human Help\n Step: {} Predicted: {} Uncertainty: {}".format(global_step,policy,ue))
            if k==ord('w'):
                human+=1
                local_step+=1
                ue_avg+=ue
                global_step+=1
                action=0
                print('phase1')
                phase1(robot)
                agent.store(s,action)
                agent.train()
            elif k==ord('e'):
                human+=1
                local_step+=1
                ue_avg+=ue
                global_step+=1
                action=N-2
                print('phase3')
                phase3(robot)
                agent.store(s,action)
                agent.train()
            elif k==ord('r'):
                human+=1
                local_step+=1
                ue_avg+=ue
                global_step+=1
                action=N-1
                print('phase4')
                phase4(robot)
                agent.store(s,action)
                agent.train()
            elif 1<int(k)-48 and int(k)-48<10 :
                human+=1
                local_step+=1
                ue_avg+=ue
                global_step+=1
                ID=int(k)-48
                action=ID-1
                left = robot.is_left(ID)
                print(ID,left)
                phase2(robot,ID,left)
                agent.store(s,action)
                agent.train()
        else:
            global_step+=1
            local_step+=1
            ue_avg+=ue
            print("Step: {} Predicted: {} Uncertainty: {}".format(global_step,policy,ue))
            agent.train()
            if policy==0:
                phase1(robot)
            elif policy==N-2:
                phase3(robot)
            elif policy==N-1:
                phase4(robot)
            else:
                left = robot.is_left(policy+1)
                phase2(robot,policy+1,left)
        if robot.goal():
            if human==0:
                flag+=1
            HUMAN.append(human/local_step)
            ue_avg=ue_avg/local_step
            UE.append(ue_avg)
            break
        
    time.sleep(1)
    robot.reset()     
agent.save() 
total_episodes=len(UE)
x=[i for i in range(total_episodes)]
plt.figure(figsize=(10,10))
plt.suptitle("4 obstacles",fontsize=20)
plt.subplot(2,1,1)
plt.xticks(x)
plt.title("avarage uncertainty")
plt.xlabel("episodes")
plt.plot(UE)
plt.tight_layout()
plt.subplot(2,1,2)
plt.xticks(x)
plt.title("humman rate")
plt.xlabel("episodies")
plt.plot(HUMAN)
plt.tight_layout()
plt.savefig("./res/res.png")
