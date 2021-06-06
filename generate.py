import random
import math
import numpy as np
# [-0.4,-0.8].[-0.3,0.3]
random.seed(0)
class BOX():
    def __init__(self,N):
        self.res_x=[]
        self.res_y=[]
        self.N=N

    def generate(self):
        self.res_x=[]
        self.res_y=[]
        x=-0.8
        y=0
        self.res_x.append(x)
        self.res_y.append(y)
        while len(self.res_x)<self.N:
            cx,cy=False,False
            while cx==False:
                x =-0.9+0.4*random.random()
                cx = self.collision_check(self.res_x,x)
            while cy==False:
                y = -0.15+0.3*random.random()
                cy = self.collision_check(self.res_y,y)
            self.res_x.append(x)
            self.res_y.append(y)
        # SORTX=np.asarray(self.res_x[1:])
        # INDEX = np.argsort(SORTX)
        # SORTX=np.sort(SORTX)
        # SORTY=np.zeros(9)
        print(self.res_x,self.res_y)

    def collision_check(self,l,x):
        for i in l:
            dis=self.l2(i,x)
            if dis<0.032:
                return False
        return True

    def l2(self,x1,x2):
        return math.sqrt((x1-x2)**2)