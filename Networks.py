import os 
import torch.nn as nn
from torch.optim import NAdam
import MinkowskiEngine as ME
import numpy as np
# from Object_v2 import rand_object, Cylinder
import torch
from time import time 
from spare_tnsr_replay_buffer import ReplayBuffer

class Actor(ME.MinkowskiNetwork):

    def __init__(self, in_feat, n_actions, D, name, chckpt_dir = 'tmp'):
        super(Actor, self).__init__(D)
        self.name = name 
        self.file_path = os.path.join(chckpt_dir,name+'_ddpg')
        conv_out1 = 32
        conv_out2 = 128
        conv_out3 = 128
        conv_out4 = 128
        layer1_unit = 512
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_feat,
                out_channels=conv_out1,
                kernel_size=3,
                stride=3,
                bias=False,
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
                bias=False,
                dimension=D),
            # ME.MinkowskiBatchNorm(conv_out4),
            ME.MinkowskiSELU()
        )
        self.pooling = ME.MinkowskiGlobalMaxPooling()
        self.linear = nn.Sequential(
            nn.Linear(conv_out4+3, layer1_unit),  # plus 3 for jnt_err 
            nn.BatchNorm1d(layer1_unit)
        )
        self.out = nn.Sequential(
            nn.Linear(layer1_unit,n_actions),
            nn.Tanh()
        )

        self.optimizer = NAdam(self.parameters())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def to_dense_tnsr(self, x:ME.SparseTensor):
        y = torch.zeros_like(x.features)
        for c in x.coordinates:
            y[int(c[0])] = x.features[int(c[0])]
        return y

    def to_3D(self, x:ME.SparseTensor):
        new_coords = torch.cat((torch.reshape(x.coordinates[:,0],(x.coordinates.shape[0],1)),x.coordinates[:,2:5]),dim=1)
        y = ME.SparseTensor(features=x.features,coordinates=new_coords,device='cuda')
        return y

    def preprocessing(self,state,single_value=False):
        if single_value:
            coords,feats = ME.utils.sparse_collate([state[0]],[state[1]])
            jnt_err = torch.tensor(state[2],dtype=coords.dtype,device='cuda').view(1,state[2].shape[0])
            # torch.tensor(state[2],device='cuda',dtype=torch.float32).view(1,state[2].shape[0])
        else: 
            coords,feats = ME.utils.sparse_collate(state[0],state[1])
            jnt_err = torch.tensor(state[2],dtype=coords.dtype,device='cuda')
            # torch.tensor(state[2],device='cuda',dtype=torch.float32)

        x = ME.SparseTensor(coordinates=coords, features=feats,device='cuda')
        return x,jnt_err

    def forward(self,x:ME.SparseTensor,jnt_err):
        print('input',x.features.shape)
        x = self.conv1(x)
        print('conv1', x.features.shape)
        x2 = self.conv2(x)
        print('conv2', x2.features.shape)
        y = self.conv3(x2)
        print('conv3',y.features.shape)
        y = self.conv4(y)
        print('conv4',y.features.shape)
        # x = self.pooling(x)
        # x = self.to_dense_tnsr(x)
        # x = torch.cat((x,jnt_err.cuda()),dim=1)
        # x = self.linear(x)
        # x = self.out(x)
        return y

    def save_checkpoint(self):
        # print('...saving ' + self.name + '...')
        torch.save(self.state_dict(), self.file_path)

    def load_checkpoint(self):
        print('...loading ' + self.name + '...')
        self.load_state_dict(torch.load(self.file_path))

class Critic(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_dim, D, name, chckpt_dir='tmp'):
        super(Critic, self).__init__(D)
        self.name = name 
        self.file_path = os.path.join(chckpt_dir,name+'_ddpg')
        conv_out1 = 64
        conv_out2 = 256
        conv_out3 = 256
        conv_out4 = 1024
        layer1_unit = 512
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_feat,
                out_channels=conv_out1,
                kernel_size=6,
                stride=3,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(conv_out1),
            ME.MinkowskiSELU()
        )
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=conv_out1,
                out_channels=conv_out2,
                kernel_size=4,
                stride=4,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(conv_out2),
            ME.MinkowskiSELU()
        )
        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=conv_out2,
                out_channels=conv_out3,
                kernel_size=(0,4,4,5),
                stride=2,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(conv_out3),
            ME.MinkowskiSELU()
        )
        self.conv4 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=conv_out3,
                out_channels=conv_out4,
                kernel_size=2,
                stride=2,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(conv_out4),
            ME.MinkowskiSELU()
        )
        self.pooling = ME.MinkowskiGlobalMaxPooling()
        self.linear = nn.Sequential(
            nn.Linear(conv_out4+6, layer1_unit),  # plus 6 for jnt_err and action
            nn.BatchNorm1d(layer1_unit)
        )
        self.out = nn.Sequential(
            nn.Linear(layer1_unit,out_dim),
        )

        self.optimizer = NAdam(self.parameters())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def to_dense_tnsr(self, x:ME.SparseTensor):
        y = torch.zeros_like(x.features)
        for c in x.coordinates:
            y[int(c[0])] = x.features[int(c[0])].clone()
        return y

    def preprocessing(self,state,action,single_value=False):
        if single_value:
            coords,feats = ME.utils.sparse_collate([state[0]],[state[1]])
            jnt_err = torch.tensor(state[2],dtype=coords.dtype,device='cuda').view(1,state[2].shape[0])
            # torch.tensor(state[2],device='cuda',dtype=torch.float32).view(1,state[2].shape[0])
        else: 
            coords,feats = ME.utils.sparse_collate(state[0],state[1])
            jnt_err = torch.tensor(state[2],dtype=coords.dtype,device='cuda')
            # torch.tensor(state[2],device='cuda',dtype=torch.float32)

        x = ME.SparseTensor(coordinates=coords, features=feats,device='cuda')
        action = torch.tensor(action,dtype=coords.dtype,device='cuda')
        return x,jnt_err,action
        

    def forward(self,x,jnt_err,action):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.pooling(x4)
        x6 = self.to_dense_tnsr(x5)
        y = torch.cat((x6,jnt_err,action),dim=1)
        x7 = self.linear(y)
        out = self.out(x7)
        return out

    def save_checkpoint(self):
        # print('...saving ' + self.name + '...')
        torch.save(self.state_dict(), self.file_path)

    def load_checkpoint(self):
        print('...loading ' + self.name + '...')
        self.load_state_dict(torch.load(self.file_path))



input = torch.ones((1,1,6,120,120,90))
x = ME.to_sparse(input,device='cuda',format='BCXXXX')


actor = Actor(1,3,D=4,name='actor')
y = actor.to_3D(x)
print(y.coordinates.size())
print(x.coordinates.size())

y = actor.forward(x,torch.ones((1,3),device='cuda'))

# print(torch.sum(y.coordinates[:,0] - x2.coordinates[:,0]))
# print(torch.sum(y.coordinates[:,1] - x2.coordinates[:,2]))
# print(torch.sum(y.coordinates[:,2] - x2.coordinates[:,3]))
# print(torch.sum(y.coordinates[:,3] - x2.coordinates[:,4]))

