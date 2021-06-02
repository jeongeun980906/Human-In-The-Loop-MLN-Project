import os
import time
import pdb
import pybullet as p
import pybullet_data
import utils_ur5_robotiq140
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt

serverMode = p.GUI # GUI/DIRECT
sisbotUrdfPath = "./urdf/ur5_robotiq_140.urdf"

# connect to engine servers
physicsClient = p.connect(serverMode)
# add search path for loadURDFs
p.setAdditionalSearchPath(pybullet_data.getDataPath())
#p.getCameraImage(640,480)

# define world
#p.setGravity(0,0,-10) # NOTE
planeID = p.loadURDF("plane.urdf")

# define environment
deskStartPos = [0.1, -1.09, 0.65]
deskStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("./urdf/objects/block.urdf", deskStartPos, deskStartOrientation)

tableStartPos = [0.0, -1.0, 0.6]
tableStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId1 = p.loadURDF("./urdf/objects/table.urdf", tableStartPos, tableStartOrientation,useFixedBase = True)
width = 128
height = 128
fov = 40
aspect = width / height
near = 0.2
far = 2
view_matrix = p.computeViewMatrix([0.0, 1.5, 0.5], [0, 0, 0.7], [0, 1, 0])
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

robotStartPos = [0,0,0.0]
robotStartOrn = p.getQuaternionFromEuler([0,0,0])
print("----------------------------------------")
print("Loading robot from {}".format(sisbotUrdfPath))
robotID = p.loadURDF(sisbotUrdfPath, robotStartPos, robotStartOrn,useFixedBase = True,
                     flags=p.URDF_USE_INERTIA_FROM_FILE)
joints, controlRobotiqC2, controlJoints, mimicParentName = utils_ur5_robotiq140.setup_sisbot(p, robotID)
eefID = 7 # ee_link
joint_init = [-1.5685099733450005, -0.685728715700564, 1.7533421524196553, -1.0673551126480323, 1.5723612330823589, 1.5715930831377436, 8.892722271385826e-07, -5.785685288437526e-09, -4.658869796347235e-09, 2.3650918950785152e-06, -6.134919914904492e-09, 2.2041008410160062e-08]

gripper_opening_length = 0.085
gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)  
for i,name in enumerate(controlJoints):
    if name==mimicParentName:
            controlRobotiqC2(controlMode=p.POSITION_CONTROL, targetPosition=gripper_opening_angle)
    else:
        p.resetJointState(robotID,joints[name].id,targetValue=joint_init[i],targetVelocity=0)
while True:
    state = p.getLinkState(robotID, eefID)
    print(state)