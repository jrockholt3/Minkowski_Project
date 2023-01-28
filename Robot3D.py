#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import math as m
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Goal is to create a joint space that the robot can operate in. As long as a decision doesn't put it out of this joint space it can go there. This could be constrain by the orientation of the part.   

# In[2]:


# settings
links = np.array([0,.3,.3])
workspace = np.sum(links)
max_obj_vel = workspace/3 # takes 3 seconds to move across the entire workspace
workspace_limits = np.array([[-workspace, workspace],[-workspace, workspace],[0, .9]])
res = 0.01
# dt = .01


# In[3]:


def c(x):
    return m.cos(x)
def s(x):
    return m.sin(x)

def T_ji(thj, aij, lij, Sj):
    return np.array([[       c(thj),        -s(thj),        0,        lij],
                     [s(thj)*c(aij),  c(thj)*c(aij),  -s(aij),  -s(aij)*Sj],
                     [s(thj)*s(aij),  c(thj)*s(aij),   c(aij),   c(aij)*Sj], 
                     [      0,             0,             0,             1]])

def T_inverse(T):
    R = T[0:3,0:3].T
#     print(R)
    shift = T[0:3,3]
#     print('shift ' + str(shift))
    shift = -R@shift
#     print('-shift ' + str(shift))
    new_T = np.zeros((4,4))
    for i in range(0,3):
        for j in range(0,3):
            new_T[i,j] = R[i,j]
    for i in range(0,3):
        new_T[i,3] = shift[i]
    new_T[3,3] = 1
    
#     print(new_T)
    return new_T

def T_ij(thj, aij, lij, Sj):
    T = T_ji(thj, aij, lij, Sj).T
    shift = np.array([-c(thj)*lij, s(thj)*lij, -Sj, 1])
    T[3,0:2] = 0
    T[:,3] = shift[:]
    return T

def T_1F(ph, S):
    return np.array([[c(ph), -s(ph), 0, 0],
                     [s(ph),  c(ph), 0, 0],
                     [    0,      0, 1, S], 
                     [    0,      0, 0, 1]])

def quantize(arr, res=res, workspace_limits=workspace_limits):
    range_ = np.abs(workspace_limits[:,1] - workspace_limits[:,0])
    ndx_range = range_/res 
    ndx = ndx_range * (arr - workspace_limits[:,0]) / range_
    return ndx   

    
# In[4]:

class robot_3link():
    def __init__(self):
        self.base = np.array([0,0,0,1])
        self.links = np.array([0,.3,.3])
        self.aph = np.array([np.pi/2, 0.0, 0.0]) # twist angles
        self.pos = np.array([0.0, 0.0, 0.0]) # joint angles
        self.S = np.array([0.3, 0.0, 0.0]) # joint offset
        self.P_3 = np.array([self.links[2],0,0,1]) # tool point as seen by 3rd coord sys
        self.v_lim = np.array([m.pi, m.pi, m.pi]) # joint velocity limits 
        self.jnt_vel = np.array([0.0, 0.0, 0.0])
        self.traj = np.array([])
#        self.th_lim = np.array()

    def set_pose(self,  th_arr):
        c = np.cos(th_arr)
        s = np.sin(th_arr)

        self.pos = np.arctan2(s,c)
        
    def get_pose(self):
        return self.pos
    
    def set_jnt_vel(self, vel_arr):
        self.jnt_vel = vel_arr
        
    def asb_bodyframe(self, scene_obj):
        T_dict = self.get_transform(self.pos)
        keys = T_dict.keys()
        obj_pos = scene_obj.curr_pos # need to change later to reflect an object's volume
        vec = np.array([obj_pos[0],obj_pos[1],0,1])
        temp = np.ones((4,3))
        for i,k in enumerate(keys):
            T = T_dict[k]
            T = T_inverse(T)
            vec_T = T@vec # pos of object in frame pov
            temp[:,i] = vec_T # gives the relative position of the vector
        
        w = self.jnt_vel
        th = self.pos
        vp_f = np.array([scene_obj.vel[0], scene_obj.vel[1],0])
        
        return temp
    
    def proximity(self, scene_obj):
        obj_asb_bodies = self.asb_bodyframe(scene_obj)
        prox_arr = np.zeros((len(self.links)))
        # body 1
        for i in range(len(self.links)):
            obj_pos = obj_asb_bodies[0:3,i]
            if obj_pos[0] <= 0: # if obj is behind ith joint
                prox1 = np.linalg.norm(obj_pos)
            elif obj_pos[0] >= self.links[0]: # if obj is past the jth joint
                prox1 = np.linalg.norm(obj_pos)
            else:
                prox1 = abs(obj_pos[1])
                
            prox_arr[i] = prox1
            
        return prox_arr
    

    # needs revision - doesn't correctly calculate relative vel
    def relative_velocity(self, scene_obj, w=None, th=None):
        if np.any(w==None): w = self.jnt_vel
        if np.any(th==None): th = self.pos
        vp_f = np.array([scene_obj.vel[0], scene_obj.vel[1], 0])
#         print('vp_f ' + str(vp_f))
        rj_f = self.forward(th=th)
        vp_j = []
        for i in range(len(self.links)):
            wj_f = np.array([0,0,np.sum(w[0:i+1])])
            vp_j.append(vp_f - np.cross(wj_f, rj_f[0:3,i]))
            
        vp_j = np.array(vp_j).T
        vp_j2 = []
        transfroms = self.get_transform(th=th)
        keys = transfroms.keys()
        for i,k in enumerate(keys):
            T = T_inverse(transfroms[k])
            vi = np.hstack((vp_j[:,i], 0))
            vp_j2.append(T@vi)
        
        
        return np.array(vp_j2).T
    
    def forward(self, th=None, make_plot=False):
        if np.any(th==None): th = self.pos
        l = self.links
        a = self.aph
        S = self.S 
        temp = np.vstack((self.base, 
                         T_1F(th[0],self.S[0])@T_ji(th[1],a[0],l[0],S[1])@np.array([0,0,0,1]),
                         T_1F(th[0],self.S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])@np.array([0,0,0,1]),
                         T_1F(th[0],self.S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])@self.P_3))
        if make_plot == True:
            self.plot_pose(arr=temp)
        return temp.T

    def state(self, obj):
        th = self.pos
        w = self.jnt_vel
        l = self.links
        a = self.aph
        S = self.S
        r_f = np.array([obj.curr_pos[0], obj.curr_pos[1], 0, 1])
        v_f = np.array([obj.vel[0], obj.vel[1], 0, 0])
        T = T_1F(th[0],self.S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])
        
        r_ee = T_inverse(T)@r_f

        dth = .01*np.pi/180
        vec = T@self.P_3
        drdth1 = (T_1F(th[0]+dth,self.S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])@self.P_3 - T_1F(th[0]-dth)@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])@self.P_3) / (2*dth)
        drdth2 = (T_1F(th[0],self.S[0])@T_ji(th[1]+dth,a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])@self.P_3 - T_1F(th[0])@T_ji(th[1]-dth,a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])@self.P_3) / (2*dth)
        drdth3 = (T_1F(th[0],self.S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2]+dth,a[1],l[1],S[2])@self.P_3 - T_1F(th[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2]-dth,a[1],l[1],S[2])@self.P_3) / (2*dth)
        J = np.vstack([drdth1, drdth2, drdth3]).T

        ee_v = J@w
        rel_vel = v_f - ee_v
        rel_vel = T_inverse(T)@rel_vel

        return np.array([r_ee[0], r_ee[1]]), np.array([rel_vel[0], rel_vel[1]])

    def get_transform(self, th=None):
        if np.any(th==None): th = self.pos
        l = self.links
        a = self.aph
        S = self.S
        dict = {'1toF':T_1F(th[0],self.S[0]), # 1st body
                '2toF':T_1F(th[0],self.S[0])@T_ji(th[1],a[0],l[0],S[1]), # second body 
                '3toF':T_1F(th[0],self.S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])} # third body
        return dict
     
    
    def reverse(self, goal, make_plot=False):
        x = goal[0]
        y = goal[1]
        z = goal[2]
        
        # th1 is the angle from x-axis of a12 in the positive RH sense. two solutions
        th1 = np.array([m.atan2(y,x), m.atan2(y,x)])
        s1 = np.sin(th1)
        c1 = np.cos(th1)
        th1 = np.arctan2(s1,c1)

        r = np.sqrt(x**2 + y**2)
        z = z - self.S[0]
        alpha = np.arctan2(z,r)
        
        c = self.links[1]
        a = self.links[2]
        b = np.sqrt(r**2 + z**2)
        
        A = np.arccos((a**2 - b**2 - c**2) / (-2*b*c)) # cosine law
        th2 = np.array([A + alpha, alpha - A])
        
        val = (b**2 - a**2 - c**2) / (-2*a*c)
        if abs(np.round(val,5)) == 1:
            th3 = np.array([0,0])
        else:
            B = np.arccos(val)
            th3 = np.array([B-np.pi, np.pi-B])

        th = np.vstack((th1, th2, th3))
        if make_plot == True: 
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            sol1 = self.forward(th[:,0])
            xx = sol1[0,:]
            yy = sol1[1,:]
            zz = sol1[2,:]
            ax.plot3D(xx,yy,zz,'red', label='Sol1')
            ax.scatter3D(xx,yy,zz,'r')
            sol1 = self.forward(th[:,1])
            xx = sol1[0,:]
            yy = sol1[1,:]
            zz = sol1[2,:]
            ax.plot3D(xx,yy,zz,'blue', label='Sol2')
            ax.scatter3D(xx,yy,zz,'b')
            ax.axes.set_xlim3d(left=-workspace, right=workspace) 
            ax.axes.set_ylim3d(bottom=-workspace, top=workspace) 
            ax.axes.set_zlim3d(bottom=0, top=workspace+self.S[0])
            ax.legend()
            plt.show()
            
        return th
    
    def plot_pose(self, th=None, arr = None):
        if np.any(th==None) and np.any(arr==None): 
            th = self.pos
            arr = self.forward(th=th)
        elif np.any(arr==None) and np.all(th!=None):
            arr = self.forward(th=th)
            
        xx = arr[:,0]
        yy = arr[:,1]
        zz = arr[:,2]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xx,yy,zz)
        ax.plot3D(xx,yy,zz,'red')
        ax.axes.set_xlim3d(left=-workspace, right=workspace) 
        ax.axes.set_ylim3d(bottom=-workspace, top=workspace) 
        ax.axes.set_zlim3d(bottom=0, top=workspace+self.S[0]) 
        plt.show()
        


# In[5]:


class rand_object():
    def __init__(self,object_radius=.03):
        self.radius = object_radius
        # init the object's location at a random (x,y) within the workspace
        rho = np.random.rand(2)*workspace/2 + workspace/4
        phi = 2*m.pi*np.random.rand()
        phi2 = (m.pi/2)*np.random.rand() - m.pi/4
        z_range = np.sum(np.abs(workspace_limits[2,:]))
        z_min = np.abs(workspace_limits[2,0])
        self.start = np.array([rho[0]*m.cos(phi), rho[0]*m.sin(phi), np.random.rand()*z_range - z_min])
        self.goal = np.array([rho[0]*m.cos(phi+phi2), rho[0]*m.sin(phi+phi2), np.random.rand()*z_range - z_min])
        v_vec = (self.goal - self.start) / np.linalg.norm((self.goal - self.start))
        self.vel = (.5*np.random.rand()+.5)*max_obj_vel*v_vec
        self.tf = np.sqrt((self.goal[0]-self.start[0])**2 + (self.goal[1]-self.start[1])**2) / np.linalg.norm(self.vel)
        self.curr_pos = self.start

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

    def step(self, time_step):
        self.curr_pos = self.curr_pos + self.vel*time_step
    
    def get_coord_list(self, res = .01, make_plot=False):
        check = [[0,res,-res],[0,res,0],[0,res,res],[0,0,res],[0,-res,res],[0,-res,0],[0,-res,-res],[0,0,-res],[0,res,-res]]
        r_int = int(np.round(self.radius/res))
        r = self.radius
        pos = self.curr_pos
        x_c,y_c = pos[0],pos[1]
        pos = pos - np.array([r,0,0])
        coord_list = []
        feat_list = []
        coord_list.append(pos)
        feat_list.append(1.0)
        pos = pos + np.array([res,0,0])
        x = pos[0]
        for i in range(-r_int+1,r_int):
            y_start = np.sqrt(abs(r**2 - (x-x_c)**2)) + y_c
            start = np.array([pos[0],y_start,pos[2]])
            pos = start
            done = False
            n = 0
            while not done:
                n += 1
                found_next = False
                for j in range(1,len(check)):
                    check_loc = pos + check[j] - self.curr_pos
                    prev_loc = pos + check[j-1] - self.curr_pos
                    if np.round(np.linalg.norm(check_loc),2) == np.round(r,2) and np.round(np.linalg.norm(prev_loc),2) > np.round(r,2):
                        coord_list.append(pos+check[j])
                        feat_list.append(1.0)
                        pos = pos + check[j]
                        j = len(check) + 1
                        found_next = True
                
                if np.all(np.round(start,2) == np.round(pos,2)) and found_next:
                    done = True
                elif not found_next:
                    done = True
                elif n > 1000:
                    done = True
            
            pos = np.array([pos[0]+res, self.curr_pos[1], self.curr_pos[2]])
            x = pos[0]
        
        coord_list.append(self.curr_pos+np.array([r,0,0])) 
        feat_list.append(1.0) 

        if make_plot:
            coord_list = np.array(coord_list)
            xx = coord_list[:,0]
            yy = coord_list[:,1]
            zz = coord_list[:,2]
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot3D(xx,yy,zz)
            ax.axes.set_xlim3d(left=self.curr_pos[0] - 1.5*r, right=self.curr_pos[0] + 1.5*r) 
            ax.axes.set_ylim3d(bottom=self.curr_pos[1] - 1.5*r, top=self.curr_pos[1] + 1.5*r) 
            ax.axes.set_zlim3d(bottom=self.curr_pos[2] - 1.5*r, top=self.curr_pos[2] + 1.5*r)
            plt.show()

        return coord_list, feat_list

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
        
    def path(self, t,set_new_pos=False):
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
        

# In[7]:


# robot = robot_3link()
# robot.forward(th=np.array([0,m.pi/2,0]), make_plot=True)
# plt.show()


# In[8]:


# obj = defined_object(np.array([0,.5]), np.array([.5,.5]), .3)


# In[9]:


# obj.vel


# In[10]:


# th_arr = np.array([0,m.pi,0])
# robot.forward(th=th_arr, make_plot=True)
# rel_v_arr = robot.relative_velocity(obj, w=np.array([0,0,0]), th=th_arr)
# np.round(rel_v_arr,4)


# In[11]:


# rel_v_arr[:,0]


# In[ ]:





# In[12]:


# T_dict = robot.get_transform(th=th_arr)
# T_Fto1 = T_inverse(T_dict['2toF'])
# v1 = rel_v_arr[:,1]
# v1 = np.hstack((v1, 0))
# print(v1)
# np.round(T_Fto1@v1,3)


# In[13]:


# prox = robot.proximity(obj)
# prox


# In[14]:


# th_arr_goal = robot.reverse(np.array([.6,.2]), m.pi/1.5, make_plot=True)
# np.round(th_arr_goal,3)


# In[15]:


# arr = np.round(robot.forward(th_arr_start[:,0]),3)
# arr[0:4,0], arr


# In[16]:


# traj = simple_trajectory(th_arr_start[:,1], th_arr_goal[:,0],robot)


# In[17]:


# plot_traj(traj, robot);


# In[18]:


# x_axis = np.array([1,0,0])
# pos = np.array([.5,-1,0])
# np.linalg.norm(np.cross(x_axis, pos)), np.dot(x_axis,pos)

