import os 
import torch.nn as nn
from torch.optim import NAdam
import MinkowskiEngine as ME
import numpy as np
from Object import rand_object, Cylinder
import torch
from time import time 
from spare_tnsr_replay_buffer import ReplayBuffer

class Actor(ME.MinkowskiNetwork):

    def __init__(self, in_feat, n_actions, D, name, chckpt_dir = 'tmp'):
        super(Actor, self).__init__(D)
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
                kernel_size=4,
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

    def forward(self,state,single_value=False):
        if single_value:
            coords,feats = ME.utils.sparse_collate([state[0]],[state[1]])
            jnt_err = torch.tensor(state[2],device='cuda',dtype=torch.float32)
            jnt_err = jnt_err.view(1, jnt_err.shape[0])
        else: 
            coords,feats = ME.utils.sparse_collate(state[0],state[1])
            jnt_err = torch.tensor(state[2],device='cuda',dtype=torch.float32)
        x = ME.SparseTensor(coordinates=coords, features=feats, device='cuda')
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pooling(x)
        x = self.to_dense_tnsr(x)
        y = torch.concat((x,jnt_err),dim=1)
        x = self.linear(y)
        x = self.out(x)
        return x

    def save_checkpoint(self):
        print('...saving ' + self.name + '...')
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
                kernel_size=4,
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
            y[int(c[0])] = x.features[int(c[0])]
        return y

    def forward(self,state,action,single_value=False):
        if single_value:
            coords,feats = ME.utils.sparse_collate([state[0]],[state[1]])
            jnt_err = torch.tensor(state[2],device='cuda',dtype=torch.float32)
            jnt_err = jnt_err.view(1, jnt_err.shape[0])
        else: 
            coords,feats = ME.utils.sparse_collate(state[0],state[1])
            jnt_err = torch.tensor(state[2],device='cuda',dtype=torch.float32)
        x = ME.SparseTensor(coordinates=coords, features=feats, device='cuda')
        action = action.cuda()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pooling(x)
        x = self.to_dense_tnsr(x)
        y = torch.concat((x,jnt_err,action),dim=1)
        x = self.linear(y)
        x = self.out(x)
        return x

    def save_checkpoint(self):
        print('...saving ' + self.name + '...')
        torch.save(self.state_dict(), self.file_path)

    def load_checkpoint(self):
        print('...loading ' + self.name + '...')
        self.load_state_dict(torch.load(self.file_path))

# memory = ReplayBuffer(int(1e3),3,6)
# jnt_err = torch.zeros(3)
# action = torch.zeros(3)
# reward = 0

# i = 0
# while i < 200:
#     obj = rand_object()
#     done = False
#     n = 0
#     while not done:
#         n += 1
#         coords, feats = obj.get_coord_list()
#         state = (coords, feats, jnt_err, obj.t)
#         obj.step()
#         coords2, feats2 = obj.get_coord_list()
#         new_state = (coords2, feats2, jnt_err, obj.t)

#         if obj.t*obj.dt >= obj.tf:
#             done = True
#             # print('broke on episode end')
#         elif n > 1000:
#             print('broke on over-run')
#             done = True
#         else:
#             done = False

#         if coords.size == 0 or coords2.size == 0:
#             print('empty array stored')
#         memory.store_transition(state, action, reward, new_state, done)
#         i = i + 1

# state, action, reward, new_state, done = memory.sample_buffer(128)
# coords = state[0]
# feats = state[1]

# # coord_list = torch.IntTensor(coord_list)
# # feat_list = torch.FloatTensor(feat_list)
# # coords = np.vstack([out1,out2,out3,out4,out5,out6,out7,out8])
# # feats = np.vstack([feat1,feat2,feat3,feat4,feat5,feat6,feat7,feat8])
# coords,feats = ME.utils.sparse_collate([coords],[feats])

# input = ME.SparseTensor(feats,coords)
# input.float()
# # print(input.dtype)
# net = Critic(1, D=4)
# out = net.forward(input)
# print(input.coordinates)
# print(out.coordinates)