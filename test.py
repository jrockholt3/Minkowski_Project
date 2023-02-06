# import Robot3D
import numpy as np
from Robot3D import robot_3link as Robot
from Object import rand_object
from Robot3D import workspace_limits as lims
from Robot3D import workspace
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# import torch 
# import MinkowskiEngine as ME
from helpers import quantize
from spare_tnsr_replay_buffer import ReplayBuffer
from time import time
import torch
import torch.nn as nn
import MinkowskiEngine as ME

# memory = ReplayBuffer(int(1e3),3,50)
# jnt_err = np.zeros(3)
# action = jnt_err.copy()
# reward = 0

# obj = Object()
# t = 0
# n=100
# for i in range(n):
#     t1 = time()
#     coord_list,feat_list = obj.get_coord_list()
#     obj.step
#     t += time() - t1
# print('method 1 t =', 1/(t/n))
# print(feat_list.shape)

# xx,yy,zz = obj.render()
# print(xx.shape, yy.shape, zz.shape)
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.plot_surface(xx,yy,zz)

# ax.set_xlim3d(lims[0,:])
# ax.set_xlabel('X')

# ax.set_ylim3d(lims[1,:])
# ax.set_ylabel('Y')

# ax.set_zlim3d(lims[2,:])
# ax.set_zlabel('Z')
# plt.show()

# i = 0
# while i < 200:
#     obj = Object()
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

# print(state[0][0])
# print(len(state[0]))

        


# print(np.vstack(coord_list))
# print(coord_list, feat_list)

# coord_list = torch.IntTensor(coord_list)
# feat_list = torch.FloatTensor(feat_list)
# coords, feats = ME.utils.sparse_collate([coord_list],[feat_list])
# print(coords, feats)
# print(coord_list[0].shape)
# print(feat_list)
# print(coords)
# print(feats)
# A = ME.SparseTensor(coordinates=coords, features=feats)
# print(A)

# coord_mem = np.empty(10, dtype=np.object)
# feat_mem = np.empty(10, dtype=np.object)
# for i in range(10):
#     coord_list, feat_list = obj.get_coord_list()
#     coord_mem[i] = coord_list
#     feat_mem[i] = feat_list
#     obj.step()

# print(coord_mem)
# print(feat_mem)


dense_tensor = torch.rand(3, 4, 11, 11, 11, 11)  # BxCxD1xD2x....xDN

dense_tensor.requires_grad = True

# Since the shape is fixed, cache the coordinates for faster inference

coordinates = ME.dense_coordinates(dense_tensor.shape)
print(dense_tensor.shape)

network = nn.Sequential(

    # Add layers that can be applied on a regular pytorch tensor

    nn.ReLU(),

    ME.MinkowskiToSparseTensor(coordinates=coordinates),

    ME.MinkowskiConvolution(4, 5, stride=2, kernel_size=3, dimension=4),

    ME.MinkowskiBatchNorm(5),

    ME.MinkowskiReLU(),

    ME.MinkowskiConvolutionTranspose(5, 6, stride=2, kernel_size=3, dimension=4),

    ME.MinkowskiToDenseTensor(

        dense_tensor.shape

    ),  # must have the same tensor stride.

)

for i in range(5):

    print(f"Iteration: {i}")

    output = network(dense_tensor) # returns a regular pytorch tensor

    output.sum().backward()