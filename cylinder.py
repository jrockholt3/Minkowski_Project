import numpy as np
from Robot3D import res, workspace, workspace_limits
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from time import time 
import Robot3D

class Cylinder():
    def __init__(self, r, L, res=res/2):
        self.r = r 
        self.L = L
        self.res = res
        self.cloud = self.make_cloud()

    def make_cloud(self):
        def circle_solve(x,r):
            #return the y-value of x^2+y^2=r^2
            return np.sqrt(r**2 - x**2)

        z = 0
        points = []
        while z < self.L:
            x = -self.r 
            while x <= self.r: # positive y values
                y = circle_solve(x, self.r)
                points.append(np.array([x,y,z,1]))
                x += self.res
            while x >= -self.r: # negative y values
                y = -1*circle_solve(x,self.r)
                points.append(np.array([x,y,z,1]))
                x += -1*self.res
            z += self.res
        
        return np.vstack(points).T

    def transform(self, T):
        self.cloud = T@self.cloud

    def plot_cloud(self):
        xx = self.cloud[0,:]
        yy = self.cloud[1,:]
        zz = self.cloud[2,:]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.axes.set_xlim3d(left=-workspace, right=workspace) 
        ax.axes.set_ylim3d(bottom=-workspace, top=workspace) 
        ax.axes.set_zlim3d(bottom=0, top=workspace_limits[2,1]) 
        ax.plot3D(xx,yy,zz)
        plt.show()



# t1 = time()            
body1 = Cylinder(.05,.3)
# t2 = time()
# print('init time takes ', t2 - t1)

# robot = Robot3D.robot_3link()
# robot.set_pose(np.array([0,np.pi/4,0]))
# T = robot.get_transform()
# T2 = T['1toF']
# # T2 = Robot3D.T_inverse(T2)
# t1 = time()
# body1.transform(T2)
# print('transform time takes ', time()-t1)
# body1.plot_cloud()
# print(body1.cloud.shape)

t1 = time()
dic = dict()
for i in range(body1.cloud.shape[1]):
    arr = np.round(body1.cloud[:,i],2)
    tup = (arr[0], arr[1], arr[2])
    dic[tup] = True 

coord_list = []
for k in dic.keys():
    coord_list.append(np.array([k[0],k[1],k[2]]))

coord_list = np.vstack(coord_list)
t2 = time()
print('down sampling time took', t2-t1)

# robot.forward(make_plot=True)