# import Robot3D
import numpy as np
from Robot3D import robot_3link as Robot
from Robot3D import rand_object as Object
from Robot3D import workspace_limits
from Robot3D import workspace
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# import torch 
# import MinkowskiEngine as ME
from Robot3D import quantize
from spare_tnsr_replay_buffer import ReplayBuffer
from time import time

memory = ReplayBuffer(int(1e3),3,50)
jnt_err = np.zeros(3)
action = jnt_err.copy()
reward = 0

obj = Object()
t = 0
n=100
for i in range(n):
    t1 = time()
    coord_list,feat_list = obj.get_coord_list()
    obj.step
    t += time() - t1
print('method 1 t =', 1/(t/n))
print(feat_list.shape)
obj.curr_pos = obj.start
for i in range(n):
    t1 = time()
    coord_list,feat_list = obj.get_coord_list2()
    obj.step
    t += time() - t1
print('method 2 t =', 1/(t/n))
print(feat_list.shape)

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
