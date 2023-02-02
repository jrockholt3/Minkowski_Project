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
    # this helper function takes in a 3xN array of (x,y,z) coords and
    # outputs the ndx of the coord based on a array representing the whole workspace
    # with resolution: "res"
    range_ = np.abs(workspace_limits[:,1] - workspace_limits[:,0])
    ndx_range = range_/res 
    ndx = np.round(ndx_range * (arr - workspace_limits[:,0]) / range_)
    return np.int16(ndx)   

    
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
        # plt.show()



class rand_object():
    def __init__(self,object_radius=.03,dt=.01667):
        self.radius = object_radius
        # init the object's location at a random (x,y) within the workspace
        rho = np.random.rand(2)*workspace/2 + workspace/4           
            # print(C_list)
        phi = 2*m.pi*np.random.rand()
        phi2 = (m.pi/2)*np.random.rand() - m.pi/4
        z_range = np.sum(np.abs(workspace_limits[2,:]))
        z_min = np.abs(workspace_limits[2,0])
        self.start = np.array([rho[0]*m.cos(phi), rho[0]*m.sin(phi), np.random.rand()*z_range - z_min])
        self.goal = np.array([rho[0]*m.cos(phi+phi2), rho[0]*m.sin(phi+phi2), np.random.rand()*z_range - z_min])
        v_vec = (self.goal - self.start) / np.linalg.norm((self.goal - self.start))
        self.vel = (.5*np.random.rand()+.5)*max_obj_vel*v_vec
        self.tf = np.linalg.norm(self.goal - self.start) / np.linalg.norm(self.vel)
        self.curr_pos = self.start
        self.t = 0 # an interger defining the time step 
        self.dt = dt # time between time steps
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

    def get_coord_list(self, res=res, make_plot=False):
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
                            coord_list.append(np.hstack([self.t,quantize(point)]))
                            feat_list.append(1)
                        x = x + res
                    x = self.curr_pos[0] + r_slice # reset x
                    while x >= self.curr_pos[0] - r_slice:
                        y = self.curr_pos[1] - y_solve(x,z,self.radius,self.curr_pos)
                        point = np.round(np.array([x,y,z]),2)
                        if check_range(point):
                            coord_list.append(np.hstack([self.t,quantize(point)]))
                            feat_list.append(1)
                        x = x - res
                    z = z + res
                else:
                    x_c,y_c = self.curr_pos[0], self.curr_pos[1]
                    point = np.array([x_c,y_c,z])
                    if check_range(point):
                            coord_list.append(np.hstack([self.t,quantize(point)]))
                            feat_list.append(1)
                    z = z + res
        else:
            print('object out of range')
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






    def get_coord_list2(self, res = res, make_plot=False):
        # this function creates a coordinate list of points along
        # the object's surface based on a spherical representation, the radius
        # and the resolution of the workspace
        # the function starts at one end of the sphere then iterates over the x-axis
        # at each "slice" along the x-axis, it finds points on the surfaces
        def check_range(point, limits):
            if np.all(point >= limits[:,0]) and np.all(point<=limits[:,1]):
                return True 
            else:
                return False

        coord_list = []
        feat_list = []
        if np.all(np.abs(self.curr_pos)-self.radius <= workspace_limits[:,1]):
            check = [[0,res,-res],[0,res,0],[0,res,res],[0,0,res],[0,-res,res],[0,-res,0],[0,-res,-res],[0,0,-res],[0,res,-res]]
            r_int = int(np.round(self.radius/res))
            r = self.radius
            pos = self.curr_pos
            x_c,y_c = pos[0],pos[1]
            pos = pos - np.array([r,0,0])
            if check_range(pos, workspace_limits): 
                coord_list.append(np.hstack([self.t,quantize(pos)]))
                feat_list.append(np.array([1]))
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
                            pos = pos + check[j]
                            if check_range(pos, workspace_limits):
                                # coord_list.append(quantize(pos).reshape(1,len(pos)))
                                coord_list.append(np.hstack([self.t,quantize(pos)]))
                                feat_list.append(np.array([1]))
                            j = len(check) + 1
                            found_next = True
                    
                    if np.all(np.round(start,2) == np.round(pos,2)) and found_next:
                        done = True
                    elif not found_next:
                        done = True
                    elif n > 1000:
                        done = True
                        print('get_coord_list broke on over flow')
                

                pos = np.array([pos[0]+res, self.curr_pos[1], self.curr_pos[2]])
                x = pos[0]
            
            arr = self.curr_pos+np.array([r,0,0])
            if check_range(arr,workspace_limits):
                # coord_list.append(quantize(pos).reshape(1,len(pos)))
                coord_list.append(np.hstack([self.t,quantize(arr)]))
                feat_list.append(np.array([1]))

            if make_plot:
                coord_list2 = np.array(coord_list)
                xx = coord_list2[:,1]
                yy = coord_list2[:,2]
                zz = coord_list2[:,3]
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.plot3D(xx,yy,zz)
                ax.scatter3D(xx,yy,zz,alpha=.5)
                # ax.axes.set_xlim3d(left=self.curr_pos[0] - 1.5*r, right=self.curr_pos[0] + 1.5*r) 
                # ax.axes.set_ylim3d(bottom=self.curr_pos[1] - 1.5*r, top=self.curr_pos[1] + 1.5*r) 
                # ax.axes.set_zlim3d(bottom=self.curr_pos[2] - 1.5*r, top=self.curr_pos[2] + 1.5*r)
                plt.show()
        else:
            print('object out of workspace', self.curr_pos)


        return np.array(coord_list), np.array(feat_list)


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
        

