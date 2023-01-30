import Robot3D
import numpy as np
from Robot3D import robot_3link as Robot
from Robot3D import rand_object as Object
from Robot3D import workspace_limits
from Robot3D import workspace
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import torch 
import MinkowskiEngine as ME
from Robot3D import quantize

obj = Object()
obj.curr_pos = np.array([0,0,.2])
coord_list, feat_list = obj.get_coord_list()

# coord_list = torch.IntTensor(coord_list)
# feat_list = torch.FloatTensor(feat_list)
coords, feats = ME.utils.sparse_collate([coord_list],[feat_list])
# print(coords, feats)
# print(coord_list[0].shape)
# print(feat_list)
# print(coords)
# print(feats)
A = ME.SparseTensor(coordinates=coords, features=feats)
print(A)