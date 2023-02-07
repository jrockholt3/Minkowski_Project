import numpy as np
import math as m
import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from helpers import quantize 


class rand_object():
    def __init__(self,object_radius=.03,dt=.01, res=0.01, max_obj_vel = 1.2, \
                label=-1.0,workspace_limits=np.array([[-.6,.6],[-.6,.6],[0,.9]])):
        self.radius = object_radius
        self.res = res
        self.t = 0 # an interger defining the time step 
        self.dt = dt # time between time steps
        self.label = label # classification of the type of object

        # init the object's location at a random (x,y) within the workspace
        self.workspace_limits = workspace_limits
        rho = np.random.rand() * self.workspace_limits[0,1]        
        phi = 2*m.pi*np.random.rand()
        phi2 = (m.pi)*np.random.rand() - m.pi/2
        z_range = np.sum(np.abs(workspace_limits[2,:])) * .7
        z_min = np.abs(workspace_limits[2,0])
        self.start = np.array([rho*m.cos(phi), rho*m.sin(phi), np.random.rand()*z_range - z_min])
        self.goal = np.array([rho*m.cos(phi+phi2), rho*m.sin(phi+phi2), np.random.rand()*z_range - z_min])
        v_vec = (self.goal - self.start) / np.linalg.norm((self.goal - self.start))
        self.vel = (.5*np.random.rand()+.5)*max_obj_vel*v_vec
        self.tf = np.linalg.norm(self.goal - self.start) / np.linalg.norm(self.vel)
        self.curr_pos = self.start
        if not np.all(np.abs(self.start)-self.radius <= workspace_limits[:,1]):
            print('starting point out of workspace')
        if not np.all(np.abs(self.start)-self.radius <= workspace_limits[:,1]):
            print('end point is out of workspace')

    def set_pos(self, pos):
        self.curr_pos = pos
        
    def path(self, t, set_new_pos=False):
        goal = self.goal
        start = self.start
        tf = self.tf
        
        
        # letting the object always move around 
        x = t*(goal[0]-start[0])/tf + start[0]
        y = t*(goal[1]-start[1])/tf + start[1]
            
        if set_new_pos == True:
            self.set_pos(np.array([x,y]))
            
        return np.array([x,y])

    def step(self, time_step=None):
        if time_step == None:
            self.t = self.t + 1
            time_step = self.t 
        else:
            self.t = time_step

        if time_step*self.dt < self.tf:
            self.curr_pos = self.curr_pos + self.vel*self.dt
        else:
            self.curr_pos = self.goal

    def get_coord_list(self, make_plot=False, return_data=False):
        workspace_limits = self.workspace_limits

        def check_range(point, limits=workspace_limits):
            if np.all(point >= limits[:,0]) and np.all(point<=limits[:,1]):
                return True 
            else:
                return False

        def y_solve(x,z,r,pos):
            x_c,y_c,z_c = pos[0],pos[1],pos[2]
            return np.sqrt(np.abs(r**2 - (z-z_c)**2 - (x-x_c)**2))

        def r_solve(z,r,z_c):
            return np.sqrt(np.abs(r**2 - (z-z_c)**2))

        if np.all(np.abs(self.curr_pos)-self.radius <= workspace_limits[:,1]):
            coord_list = []
            feat_list = []
            z = self.curr_pos[2] - self.radius
            while z <= self.curr_pos[2] + self.radius:
                # perp distance from x-axis to sphere surface 
                r_slice = r_solve(z,self.radius,self.curr_pos[2])
                if np.round(r_slice,2) > 0:
                    x = self.curr_pos[0] - r_slice
                    while x <= self.curr_pos[0] + r_slice:
                        y = self.curr_pos[1] + y_solve(x,z,self.radius,self.curr_pos)
                        point = np.round(np.array([x,y,z]),2)
                        if check_range(point):
                            if return_data:
                                coord_list.append(torch.tensor([self.t,point]))
                                feat_list.append(self.label)
                            else:
                                coord_list.append(torch.hstack([torch.tensor([self.t],dtype=torch.float),quantize(point)]))
                                feat_list.append(self.label)
                        x = x + self.res
                    x = self.curr_pos[0] + r_slice # reset x
                    while x >= self.curr_pos[0] - r_slice:
                        y = self.curr_pos[1] - y_solve(x,z,self.radius,self.curr_pos)
                        point = np.round(np.array([x,y,z]),2)
                        if check_range(point):
                            if return_data:
                                coord_list.append(torch.tensor([self.t,point]))
                                feat_list.append(self.label)
                            else:
                                coord_list.append(torch.hstack([torch.tensor([self.t],dtype=torch.float),quantize(point)]))
                                feat_list.append(self.label)
                        x = x - self.res
                    z = z + self.res
                else:
                    x_c,y_c = self.curr_pos[0], self.curr_pos[1]
                    point = np.array([x_c,y_c,z])
                    if check_range(point):
                        if return_data:
                            coord_list.append(torch.tensor([self.t,point]))
                            feat_list.append(self.label)
                        else:
                            coord_list.append(torch.hstack([torch.tensor([self.t],dtype=torch.float),quantize(point)]))
                            feat_list.append(self.label)
                    z = z + self.res
        else:
            print('object out of range', self.curr_pos)
            return

        if make_plot:
            coord_list2 = np.array(coord_list)
            xx = coord_list2[:,0]
            yy = coord_list2[:,1]
            zz = coord_list2[:,2]
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot3D(xx,yy,zz)
            ax.scatter3D(xx,yy,zz,alpha=.5)
            ax.set_xlabel('x-axis')
            ax.set_ylabel('y-axis')
            ax.set_zlabel('z-axis')
            # ax.axes.set_xlim3d(left=self.curr_pos[0] - 1.5*r, right=self.curr_pos[0] + 1.5*r) 
            # ax.axes.set_ylim3d(bottom=self.curr_pos[1] - 1.5*r, top=self.curr_pos[1] + 1.5*r) 
            # ax.axes.set_zlim3d(bottom=self.curr_pos[2] - 1.5*r, top=self.curr_pos[2] + 1.5*r)
            plt.show()
        
        return np.vstack(coord_list), np.vstack(feat_list)

    def render(self):
        u = np.linspace(0,2*np.pi,20)
        v = np.linspace(0,np.pi,20)

        x = self.radius * np.outer(np.cos(u),np.sin(v)) + self.curr_pos[0]
        y = self.radius * np.outer(np.sin(u),np.sin(v)) + self.curr_pos[1]
        z = self.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + self.curr_pos[2]

        return x,y,z


class Cylinder():
    def __init__(self, r=.05, L=.3, res=.01/1.5, label=1.0):
        self.r = r 
        self.L = L
        self.res = res
        self.original = self.make_cloud()
        self.cloud = self.original.copy()
        self.label = label

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
        # ax.axes.set_xlim3d(left=-workspace, right=workspace) 
        # ax.axes.set_ylim3d(bottom=-workspace, top=workspace) 
        # ax.axes.set_zlim3d(bottom=0, top=workspace_limits[2,1]) 
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

    def get_coord_list(self, t):
        arr = self.down_sample()
        coord_list = []
        feat_list = []
        for i in range(arr.shape[1]):
            coord_list.append(torch.hstack([torch.tensor(t),quantize(arr[:,i],res=self.res)]))
            feat_list.append(self.label)
        return np.vstack(coord_list), np.vstack(feat_list)



'''
Currently defined object is not used and is decrepit
'''
class defined_object():
    def __init__(self, start, goal, vel, object_radius=.03):
        self.radius = object_radius
        # init the object's location at a random (x,y) within the workspace
        self.start = start
        self.goal = goal
        v_vec = (self.goal - self.start) / np.linalg.norm((self.goal - self.start))
        self.vel = vel*v_vec
        self.tf = np.sqrt((self.goal[0]-self.start[0])**2 + (self.goal[1]-self.start[1])**2) / np.linalg.norm(self.vel)
        self.curr_pos = self.start
        
    def set_pos(self, pos):
        self.curr_pos = pos
        
    def path(self, t, set_new_pos=False):
        goal = self.goal
        start = self.start
        tf = self.tf
        
        if t < tf:
            x = t*(goal[0]-start[0])/tf + start[0]
            y = t*(goal[1]-start[1])/tf + start[1]
        else:
            x = goal[0]
            y = goal[1]
            
        if set_new_pos == True:
            self.set_pos(np.array([x,y]))
            
        return np.array([x,y])
    
    def contact_point(self, object_pos):
        return -1*approach_vec*self.radius