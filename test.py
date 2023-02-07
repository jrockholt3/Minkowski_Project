import Robot3D
import numpy as np
from Robot3D import robot_3link as Robot
from Object import rand_object
from Robot3D import workspace_limits as lims
from Robot3D import workspace
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
# import torch 
# import MinkowskiEngine as ME
from helpers import quantize
from spare_tnsr_replay_buffer import ReplayBuffer
from time import time
import torch
import torch.nn as nn
# import MinkowskiEngine as ME
# from Networks import Actor

# net = Actor(1,3,D=4)

robot = Robot3D.robot_3link()
obj = rand_object()
print(robot.body2.cloud.shape)
robot.set_pose(np.array([0, np.pi/4, -1*np.pi/3]))
# robot.forward(make_plot=True)
robot_coords,robot_feat = robot.get_coord_list()
obj_coords, obj_feat = obj.get_coord_list()
print(obj_coords.shape)
# coords = torch.vstack([robot_coords,obj_coords])
# feats = torch.vstack([robot_feat, obj_feat])

# coords, feats = ME.utils.sparse_collate([coords],[feats])
# A = ME.SparseTensor(coordinates=coords, features=feats)
n = 10
t = 0
# net.eval()
for i in range(n):
    t1 = time()
    robot_coords,robot_feat = robot.get_coord_list()
    # obj_coords, obj_feat = obj.get_coord_list()
    coords = torch.vstack([robot_coords,obj_coords])
    feats = torch.vstack([robot_feat,obj_feat])
    # coords, feats = ME.utils.sparse_collate([coords],[feats])
    # A = ME.SparseTensor(coordinates=coords, features=feats, device='cuda')
    # action = net.forward(A)
    t2 = time()
    t += t2 - t1

print('freq is', n/t)
print('avg time', t/n)


# memory = ReplayBuffer(int(1e3),3,50)
# jnt_err = torch.zeros(3)
# action = jnt_err.clone()
# reward = 0

# obj = rand_object()
# # t = 0
# # n=100
# # for i in range(n):
# #     t1 = time()
# #     coord_list,feat_list = obj.get_coord_list()
# #     obj.step
# #     t += time() - t1
# # print('method 1 t =', 1/(t/n))
# # print(feat_list.shape)

# # xx,yy,zz = obj.render()
# # print(xx.shape, yy.shape, zz.shape)
# # fig = plt.figure()
# # ax = fig.add_subplot(111,projection='3d')
# # ax.plot_surface(xx,yy,zz)

# # ax.set_xlim3d(lims[0,:])
# # ax.set_xlabel('X')

# # ax.set_ylim3d(lims[1,:])
# # ax.set_ylabel('Y')

# # ax.set_zlim3d(lims[2,:])
# # ax.set_zlabel('Z')
# # plt.show()

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

# # print(state[0])
# print(state[0])

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

