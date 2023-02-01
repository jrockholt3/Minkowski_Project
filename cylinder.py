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
        self.original = self.make_cloud()
        self.cloud = self.make_cloud()

    def make_cloud(self):
        def circle_solve(x,r):
            #return the y-value of x^2+y^2=r^2
            return np.sqrt(np.abs(r**2 - x**2))

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
        self.cloud = T@self.original

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


    def down_sample(self):
        dic = dict()
        for i in range(1,self.cloud.shape[1]):
            arr = np.round(self.cloud[:,i],2)
            tup = (arr[0], arr[1], arr[2])
            dic[tup] = True

        coord_list = []
        for k in dic.keys():
            coord_list.append(np.array([k[0],k[1],k[2]]))

        return np.vstack(coord_list).T

         
# res_arr = np.linspace(1,2,100)
# ts = []
# num = []
# n = 100
# for res_i in res_arr:
#     body1 = Cylinder(.05,.3,res=res/res_i)
#     t=0
#     for i in range(n):
#         t1 = time()
#         arr2 = body1.down_sample()
#         t2 = time()
#         t += (t2-t1)
#     num.append(arr2.shape[1])
#     ts.append(t/n)

# fig = plt.figure()
# plt.plot(res_arr, ts)
# plt.xlabel('res')
# plt.ylabel('run time')
# fig2 = plt.figure()
# plt.plot(res_arr,num)
# plt.xlabel('res')
# plt.ylabel('density')
# fig3 = plt.figure()
# plt.scatter(num,ts)
# plt.xlabel('density')
# plt.ylabel('run time')
# plt.show()

body0 = Cylinder(.05,.3,res=res/1)
body1 = Cylinder(.05,.3,res=res/1)
body2 = Cylinder(.05,.3,res=res/1)
robot = Robot3D.robot_3link()
robot.set_pose(np.array([5*np.pi/4,3*np.pi/4,np.pi/4]))
temp = robot.forward()
T = robot.get_transform()
th = np.pi/2
y_rot = np.array([[np.cos(th), 0, np.sin(th),0],
                  [0, 1, 0, 0],
                  [-np.sin(th), 0, np.cos(th), 0],
                  [0, 0, 0, 1]])
T1 = T['2toF']@y_rot
T2 = T['3toF']@y_rot
# T2 = Robot3D.T_inverse(T2)
t1 = time()
body1.transform(T1)
body2.transform(T2)
# print('transform time is', time()-t1)
# t1 = time()
arr1 = body1.down_sample()
arr2 = body2.down_sample()
print('down sample time is', time()-t1)

arr0 = body0.down_sample()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(arr0[0,:], arr0[1,:], arr0[2,:],alpha=.5,s=.5)
ax.scatter3D(arr1[0,:], arr1[1,:], arr1[2,:],alpha=.5,s=.5)
ax.scatter3D(arr2[0,:], arr2[1,:], arr2[2,:],alpha=.5,s=.5)
ax.plot3D(temp[0,:],temp[1,:],temp[2,:])
ax.scatter3D(temp[0,:],temp[1,:],temp[2,:])

ax.axes.set_xlim3d(left=-workspace, right=workspace) 
ax.axes.set_ylim3d(bottom=-workspace, top=workspace) 
ax.axes.set_zlim3d(bottom=0, top=workspace_limits[2,1]) 
plt.show()
