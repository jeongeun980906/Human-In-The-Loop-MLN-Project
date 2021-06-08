import os
import time
import pdb
import pybullet as p
import pybullet_data
import env.utils_ur5
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt
from env.generate import BOX
import cv2

class robot_HIL():
    def __init__(self):
        self.serverMode = p.GUI # GUI/DIRECT
        self.sisbotUrdfPath = "./urdf/ur5_ee.urdf"

        # connect to engine servers
        self.physicsClient = p.connect(self.serverMode)
        # add search path for loadURDFs
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.load_urdf()
        # define world
        #p.setGravity(0,0,-10) # NOTE
    def load_urdf(self):
            p.setGravity(0, 0, -9.8)
            self.planeID = p.loadURDF("plane.urdf")

            # define environment
            # deskStartPos = [0.0, -0.69, 0.85]
            # deskStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
            # self.boxId1 = p.loadURDF("./urdf/objects/block.urdf", deskStartPos, deskStartOrientation)
            # self.BOX=BOX()
            # self.BOX.generate()
            self.load_boxes()
            tableStartPos = [0.0, -0.8, 0.8]
            tableStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
            self.boxId1 = p.loadURDF("./urdf/objects/table.urdf", tableStartPos, tableStartOrientation,useFixedBase = True)
            self.selfID = p.loadURDF("./urdf/objects/self.urdf", tableStartPos, tableStartOrientation,useFixedBase = True)
            self.width = 128
            self.height = 128
            self.fov = 40
            self.aspect = self.width / self.height
            self.near = 0.2
            self.far = 2
            self.view_matrix = p.computeViewMatrix([0.0, -0.3, 2.0], [0, -1.0, 0], [0, 1, 0])
            self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)

            robotStartPos = [0,0,0.0]
            robotStartOrn = p.getQuaternionFromEuler([0,0,0])
            print("----------------------------------------")
            print("Loading robot from {}".format(self.sisbotUrdfPath))
            self.robotID = p.loadURDF(self.sisbotUrdfPath, robotStartPos, robotStartOrn,useFixedBase = True,
                                flags=p.URDF_USE_INERTIA_FROM_FILE)
            self.joints, self.controlJoints = utils_ur5.setup_sisbot(p, self.robotID)
            self.eefID = 7 # ee_link
            self.joint_init = [-2.745863102373773, -1.3999988707213558, 2.341539343761223, -0.9451306171004619, 0.39671122013719795, 1.5748969180289005]

            self.gripper_opening_length = 0.085
            self.gripper_opening_angle = 0.715 - math.asin((self.gripper_opening_length - 0.010) / 0.1143)  
            self.home_pose()
    
    def load_boxes(self):
        self.boxId=[]
        deskStartPos = [0,-0.8, 0.85]
        deskStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        temp = p.loadURDF("./urdf/objects/block.urdf", deskStartPos, deskStartOrientation)  
        self.boxId.append(temp)
        for i in range(4):        
            deskStartPos = [0.03*(2*(i-1)+1),-0.6-0.1*i, 0.85]
            temp = p.loadURDF("./urdf/objects/block2.urdf", deskStartPos, deskStartOrientation)        
            self.boxId.append(temp)
    
    def get_img(self):
        projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        images = p.getCameraImage(self.width,self.height,self.view_matrix,projection_matrix,shadow=True,renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_opengl = np.reshape(images[2], (self.height, self.width, 4))
        return rgb_opengl[:,:,:3]

    def home_pose(self):
        for i,name in enumerate(self.controlJoints):
            p.resetJointState(self.robotID,self.joints[name].id,targetValue=self.joint_init[i],targetVelocity=0)
        self.ori = p.getLinkState(self.robotID,self.eefID)[1]
        p.stepSimulation()
    
    def ee_pos(self):
        state = p.getLinkState(self.robotID, self.eefID)[0]
        pos = [state[0],state[1]-0.05,state[2]]
        return pos

    def move_pos(self,pos):
        pos[1]+=0.05
        joint_states = p.calculateInverseKinematics(self.robotID,self.eefID,pos,self.ori)
        for i, name in enumerate(self.controlJoints):
            p.setJointMotorControl2(self.robotID,self.joints[name].id,p.POSITION_CONTROL,
                                                targetPosition=joint_states[i],force=self.joints[name].maxForce,
                                                maxVelocity = self.joints[name].maxVelocity)
        #print(joint_states)
        for _ in range(100):
            p.stepSimulation()     
    def reset(self):
        p.resetSimulation()
        self.load_urdf()

    def goal(self):
        ee_pos=self.ee_pos()[:2]
        goal_pos, _ = p.getBasePositionAndOrientation(self.boxId[0])
        goal_pos=list(goal_pos[:2])
        dis=self.distance(goal_pos,ee_pos)
        if dis<0.01:
            return True
        return False

    def get_path(self,thres):
        ee_pos = self.ee_pos()[:2]
        goal_pos, _ = p.getBasePositionAndOrientation(self.boxId[0])
        goal_pos=list(goal_pos[:2])
        ee_pos = np.asarray(ee_pos)
        goal_pos = np.asarray(goal_pos)
        res=0
        min_dis=100
        is_left=None
        for i in range(4):
            ID=self.boxId[i+1]
            pos,_=p.getBasePositionAndOrientation(ID)
            pos=np.asarray(list(pos[:2]))
            dis=abs(pos[0] - goal_pos[0])
            if dis<thres:
                dis_goal=np.linalg.norm(ee_pos-pos)
                if dis_goal<min_dis and goal_pos[1]<pos[1] and goal_pos[1]<(ee_pos[1]-0.05):
                    res=ID
                    min_dis=dis_goal
                    d = (pos[0] - goal_pos[0])
                    if d<0:
                        is_left=1
                    else:
                        is_left=-1
        return res,is_left

    def distance(self,x1,x2):
        res=0
        for x1,x2 in zip(x1,x2):
            #print(x1,x2)
            res+=(x1-x2)**2
        return math.sqrt(res)

    def block_pos(self,ID):
        pos,_=p.getBasePositionAndOrientation(ID)
        return list(pos)

    def state(self):
        state=[]
        for i in range(5):
            ID=self.boxId[i]
            pos,_=p.getBasePositionAndOrientation(ID)
            pos=list(pos[:2])
            state.append(pos)
        ee_pos=self.ee_pos()
        ee_pos=[[ee_pos[0],ee_pos[1]]]
        state +=ee_pos
        state=np.asarray(state)
        state[:,1]=(state[:,1]+0.9)*2
        state[:,0]=(state[:,0]+0.3)/0.6
        return state
    def marker(self):
        res=[]
        for i in range(4):
            ID=self.boxId[i+1]
            pos,_=p.getBasePositionAndOrientation(ID)
            pos=list(pos[:2])
            x=int(280*pos[0]+124)            
            y=int(-(pos[1]+0.6)*240+115)
            res.append((x,y))
        return res
    def is_left(self,id):
        pos=self.ee_pos()
        pos=list(pos[:2])
        goal_pos,_=p.getBasePositionAndOrientation(id)
        goal_pos=list(goal_pos[:2])
        print(pos,goal_pos)
        d = (pos[0] - goal_pos[0])
        if d<0:
            return -1
        else:
            return 1



if __name__=='__main__':
    robot=robot_HIL()
    for _ in range(10):
        flag=0
        cv2.namedWindow('image')
        while True:
            pos=robot.ee_pos()
            img = robot.get_img()
            print(robot.get_path(0.05))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('image',img)
            if flag==0:
                pass
            k = cv2.waitKey(1) & 0xFF
            if k==ord('w'):
                pos[1]+= 0.01
            elif k==ord('s'):
                pos[1] -=0.01
            elif k==ord('a'):
                pos[0]-=0.01
            elif k==ord('d'):
                pos[0]+=0.01
            elif k==ord('q'):
                    robot.reset()
                    break
            robot.move_pos(pos)
            print(robot.goal())            