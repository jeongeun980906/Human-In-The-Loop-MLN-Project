from MLN.mlp import MixtureLogitNetwork_mlp
from MLN.loss import mace_loss,mln_uncertainties,mln_gather
from MLN.buffer import buffer
import torch
import torch.optim as optim

device='cuda'

class Agent():
    def __init__(self,lr=1e-4,ratio=1,ratio2=1,thres=1.7,wd=1e-4,action_size=11,state_size=[10,2]):
        self.buffer=buffer()
        self.lr=lr
        self.ratio=ratio
        self.ratio2=ratio2
        self.thres=thres
        self.wd=wd
        self.action_size=action_size
        self.state_size=state_size
        self.MLN=MixtureLogitNetwork_mlp(x_dim=self.state_size,
                                    y_dim=self.action_size).to(device)
        self.optm = optim.Adam(self.MLN.parameters(),lr=self.lr,
                                    weight_decay=self.wd)
    
    def train(self,epoch=3,batch_size=3):
        STEP=int(self.buffer.len()/batch_size)
        self.buffer.sampling()
        mace=0
        alea=0
        epis=0
        total=(epoch)*(STEP)
        for e in range(epoch):
            for i in range(STEP):
                batch_in,batch_out=self.buffer.get_item(batch_size)
                mln_out = self.MLN.forward(batch_in.to(device))
                pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
                target = torch.eye(self.action_size)[batch_out].to(device)
                loss_out = mace_loss(pi,mu,sigma,target)
                loss = loss_out['mace_avg'] - self.ratio*loss_out['epis_avg'] + self.ratio2*loss_out['alea_avg']
                mace+=loss_out['mace_avg'].cpu().item()
                epis+=loss_out['epis_avg'].cpu().item()
                alea+=loss_out['alea_avg'].cpu().item()
                self.optm.zero_grad() # reset gradient
                loss.backward() # back-propagation
                self.optm.step() # optimizer update
            self.buffer.done_epoch()
        try:
            mace=mace/total
            epis=epis/total
            alea=alea/total
            print("MACE: {} epis: {} alea: {}".format(mace,epis,alea))
        except:
            pass

    def play(self,x):
        x=torch.FloatTensor(x).unsqueeze(0)
        self.MLN.eval()
        mln_out = self.MLN.forward(x.to(device))
        pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
        out = mln_gather(pi,mu,sigma)
        action=out['mu_sel']
        _,action=torch.max(action,1)
        action=action.cpu().item()
        u_out=mln_uncertainties(pi,mu,sigma)
        ue = u_out['epis']+u_out['alea']
        ue = ue.cpu().item()
        if ue>self.thres:
            return action,True,ue
        else:
            return action,False,ue

    def store(self,s,a):
        self.buffer.store(s,a)

    def save(self):
        torch.save(self.MLN.state_dict(),'./ckpt/{}.pt'.format(self.buffer.len()))

    def load(self,len):
        state_dict=torch.load('./ckpt/{}.pt'.format(len))
        self.MLN.load_state_dict(state_dict)