import os 
import torch.nn as nn
from torch.optim import NAdam
import MinkowskiEngine as ME
import numpy as np
# from Object_v2 import rand_object, Cylinder
import torch
from time import time 
from spare_tnsr_replay_buffer import ReplayBuffer
from Robot_Env import tau_max

conv_out1 = 32
conv_out2 = 128
conv_out3 = 128
conv_out4 = 512
dropout = 0.1

class Obj_Pred_Net(ME.MinkowskiNetwork):

    def __init__(self, lr, in_feat, D, name='obj_vel_pred', chckpt_dir = 'tmp',top_only=False):
        super(Obj_Pred_Net, self).__init__(D)
        self.name = name 
        self.file_path = os.path.join(chckpt_dir,name)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_feat,
                out_channels=conv_out1,
                kernel_size=3,
                stride=3,
                bias=True,
                dimension=D),
            ME.MinkowskiBatchNorm(conv_out1),
            ME.MinkowskiSELU()
        )
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=conv_out1,
                out_channels=conv_out2,
                kernel_size=2,
                stride=2,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(conv_out2),
            ME.MinkowskiSELU()
        )
        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=conv_out2,
                out_channels=conv_out3,
                kernel_size=(1,4,4,4),
                stride=(0,4,4,4),
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(conv_out3),
            ME.MinkowskiSELU()
        )
        self.conv4 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=conv_out3,
                out_channels=conv_out4,
                kernel_size=(1,5,5,4),
                stride=(0,5,5,4),
                bias=True,
                dimension=D),
            # ME.MinkowskiBatchNorm(conv_out4),
            # ME.MinkowskiSELU()
        )
        self.pooling = ME.MinkowskiGlobalAvgPooling()
        self.norm = nn.Sequential(nn.BatchNorm1d(conv_out4),nn.SELU())
        self.dropout1 = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(conv_out4,12,bias=True)
        )

        if top_only:
            trainabale_params = []
            for name,p in self.named_parameters():
                if "conv" not in name:
                    trainabale_params.append(p)
            self.optimizer = NAdam(params=trainabale_params, lr=lr)
        else:
            self.optimizer = NAdam(params=self.parameters(), lr=lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def to_dense_tnsr(self, x:ME.SparseTensor):
        y = torch.zeros_like(x.features)
        for c in x.coordinates:
            y[int(c[0])] = x.features[int(c[0])]
        return y

    def preprocessing(self,state,targ,single_value=False):
        if single_value:
            coords,feats = ME.utils.sparse_collate([state[0]],[state[1]])
            targ = torch.tensor(targ,dtype=coords.dtype,device='cuda').view(1,targ.shape[0])
            # torch.tensor(state[2],device='cuda',dtype=torch.float32).view(1,state[2].shape[0])
        else: 
            coords,feats = ME.utils.sparse_collate(state[0],state[1])
            targ = torch.tensor(targ,dtype=coords.dtype,device='cuda')
            # torch.tensor(state[2],device='cuda',dtype=torch.float32)

        x = ME.SparseTensor(coordinates=coords, features=feats,device='cuda')
        return x,targ

    def forward(self,x:ME.SparseTensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pooling(x)
        x = self.to_dense_tnsr(x)
        x = self.norm(x)
        x = self.dropout1(x)
        x = self.out(x) 
        return x

    def save_checkpoint(self):
        print('...saving ' + self.name + '...')
        torch.save(self.state_dict(), self.file_path)

    def load_checkpoint(self):
        print('...loading ' + self.name + '...')
        self.load_state_dict(torch.load(self.file_path))


# batch = 2
# input = torch.ones((batch,1,6,9,120,50))
# x = ME.to_sparse(input,device='cuda',format='BCXXXX')


# actor = Actor(1,n_actions=3, D=4,name='actor')
# y = actor.to_3D(x)
# print(y.coordinates.size())
# print(x.coordinates.size())

# y = actor.forward(x,torch.ones((batch,3),device='cuda'))

# # print(torch.sum(y.coordinates[:,0] - x2.coordinates[:,0]))
# # print(torch.sum(y.coordinates[:,1] - x2.coordinates[:,2]))
# # print(torch.sum(y.coordinates[:,2] - x2.coordinates[:,3]))
# # print(torch.sum(y.coordinates[:,3] - x2.coordinates[:,4]))

