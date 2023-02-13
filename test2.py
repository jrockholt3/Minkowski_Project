import numpy as np
from Robot_Env import RobotEnv
from Robot3D import robot_3link
from time import time 
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d

from Object_v2 import rand_object as Obj
from Object_v2 import Cylinder as Cyl

obj = Obj()
cyl = Cyl()

def c(th): return np.cos(th)
def s(th): return np.sin(th)
th = np.pi/2
Ty = np.array([[ c(th), 0.0, s(th), 0.0],
              [  0.0,  1.0,   0.0, 0.0],
              [-s(th), 0.0, c(th), 0.0],
              [   0.0, 0.0,   0.0, 1.0]])
Tx = np.array([[1.0,   0.0,  0.0, 0.0],
               [0.0, c(th),-s(th), 0.0],
               [0.0, s(th), c(th), 0.0],
               [0.0,   0.0,   0.0, 1.0]])

# vec = np.array([0.0,0.0,1.0,1.0])

# print(homo_trans(Tx,vec))

t = 0
n = 1000
for i in range(n):
    t1 = time()
    obj.get_coords()
    cyl.get_coords(0.0,Ty)
    t += time() - t1

print('numba freq', n/t)

