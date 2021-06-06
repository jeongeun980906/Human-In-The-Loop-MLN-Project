import random
import torch
class buffer():
    def __init__(self):
        self.states=[]
        self.actions=[]
        self.index=0
    
    def get_item(self,batch_size=3):
        s=[]
        a=[]
        for j in range(batch_size):
            s.append(self.states[self.p[self.index*batch_size+j]])
            a.append(self.actions[self.p[self.index*batch_size+j]])
        self.index+=1
        s=torch.FloatTensor(s)
        a=torch.tensor(a,dtype=torch.int64)
        return s,a
    
    def done_epoch(self):
        self.index=0

    def sampling(self):
        self.p=[i for i in range(len(self.actions))]
        random.shuffle(self.p)
    
    def reset(self):
        self.states=[]
        self.actions=[]
    
    def store(self,s,a):
        self.states.append(s.tolist())
        self.actions.append(a)
    def len(self):
        return len(self.actions)
