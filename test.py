import Robot3D
import numpy as np
from Robot3D import robot_3link as Robot
from Object import rand_object
from Robot3D import workspace_limits as lims
from Robot3D import workspace
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import MinkowskiEngine as ME
from helpers import quantize
from spare_tnsr_replay_buffer import ReplayBuffer
from time import time
import torch
import torch.nn as nn
from Networks import Actor
from Agent import Agent
from Robot_Env import RobotEnv
from collections import OrderedDict

# net = Actor(1,3,D=4)

robot = Robot3D.robot_3link()
obj = rand_object()
robot.set_pose(np.array([0, np.pi/4, -1*np.pi/3]))
# robot.forward(make_plot=True)
robot_coords,robot_feat = robot.get_coord_list()
obj_coords, obj_feat = obj.get_coord_list()
coords = torch.vstack([robot_coords,obj_coords])
feats = torch.vstack([robot_feat, obj_feat])

coords, feats = ME.utils.sparse_collate([coords],[feats])
A = ME.SparseTensor(coordinates=coords, features=feats)
actor = Actor(1,3,4,'actor')
actor.eval()
actor.forward_prep((obj_coords,obj_feat,torch.zeros(3)),single_value=True)

# n = 10
# t = 0
# # net.eval()
# for i in range(n):
#     t1 = time()
#     robot_coords,robot_feat = robot.get_coord_list()
#     # obj_coords, obj_feat = obj.get_coord_list()
#     coords = torch.vstack([robot_coords,obj_coords])
#     feats = torch.vstack([robot_feat,obj_feat])
#     # coords, feats = ME.utils.sparse_collate([coords],[feats])
#     # A = ME.SparseTensor(coordinates=coords, features=feats, device='cuda')
#     # action = net.forward(A)
#     t2 = time()
#     t += t2 - t1

# print('freq is', n/t)
# print('avg time', t/n)


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

# memory = ReplayBuffer(int(1e3),3,6)
# jnt_err = torch.zeros(3)
# action = jnt_err.clone()
# reward = 0

# i = 0
# while i < 200:
#     obj = rand_object()
#     robot = Robot3D.robot_3link()
#     done = False
#     n = 0
#     while not done:
#         n += 1
#         obj_coords, obj_feats = obj.get_coord_list()
#         robot_coords, robot_feats = robot.get_coord_list(obj.t)
#         coords = torch.vstack([obj_coords,robot_coords])
#         feats = torch.vstack([obj_feats, robot_feats])
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

# env = RobotEnv()
# agent = Agent(env)



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

