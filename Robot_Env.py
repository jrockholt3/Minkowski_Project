import numpy as np
import Robot3D as Robot
from math import atan2
from Object_v2 import rand_object
# import torch 

# global variables
dt = 0.016 # time step
t_limit = 5 # time limit in seconds
thres = np.array([.03, .03, .03]) # joint error threshold
vel_thres = thres # joint velocity error threshold for stopping
# weighting different parts of the reward function
Alpha = 1 # area under joint err curve
Beta = 0 # positive reward for proximity
Gamma = 0 # negative rewards for be jumps in torque (minimize jerk)
prox_thres = .05 # proximity threshold - 5 cm
goal = np.array([np.pi/2, np.pi/2, np.pi/2])
min_prox = 0.15

# Controller gains
tau_max =  10 #J*rad/s^2, J = 1
damping = tau_max*.5
P = 10.0
D = 5

jnt_vel_max = np.pi/1.5 # rad/s

rng = np.random.default_rng()

def a(list):
    return np.array(list)

# robot = Robot.robot_3link()
# pos1 = robot.reverse(a([-.6,.1]),np.pi/4)
# pos1 = pos1[:,1]
# robot.forward(th=pos1,make_plot=True)
# pos2 = robot.reverse(a([.6,.1]), 2*np.pi/3)
# pos2 = pos2[:,0]
# robot.forward(th=pos2,make_plot=True)
# pos3 = robot.reverse(a([.5,-.3]), 3*np.pi/4)
# pos3 = pos3[:,1]
# robot.forward(th=pos3,make_plot=True)
# pos4 = robot.reverse(a([-0.6,-0.6]), np.pi/3)
# pos4 = pos4[:,0]
# robot.forward(th=pos4,make_plot=True)
# XX = a([pos1.copy(), pos2.copy(), pos3.copy(), pos4.copy()])
# del robot

def calc_jnt_err(th1,th2):
    # angle error is defined in the right-hand positive sense from th1 to th2
    # th1 is the current position of the robot, th2 is the goal
    x1 = np.cos(th1)
    x2 = np.cos(th2)
    y1 = np.sin(th1)
    y2 = np.sin(th2)

    r1 = np.array([x1,y1])
    r2 = np.array([x2,y2])
    s = []
    c = []
    for i in range(r1.shape[1]):
        arr = np.cross(r1[:,i], r2[:,i])
        s.append(arr)
        c.append(np.dot(r1[:,i], r2[:,i]))  
        
    arr = []
    for i in range(len(s)):
        th = atan2(s[i],c[i])
        if np.round(th,5) == -np.round(np.pi,5):
            th = np.pi
        arr.append(th)

    # arr = jnt_err vector
    return np.array(arr) 

def angle_calc(th):
    s = np.sin(th)
    c = np.cos(th)
    return np.arctan2(s,c)

def gen_rand_pos():
    vec = (2*np.random.rand(3) - 1)
    mag = .4*np.random.rand() + .2
    goal = mag * .999 * (vec/np.linalg.norm(vec)) + np.array([0,0,.3])
    return goal 

class PDControl():
    def __init__(self, P=P, D=D, tau_max=tau_max, dt=dt, damping=damping):
        self.P = P
        self.D = D
        self.jnt_err = np.array([0,0,0])
        self.prev_jnt_err = np.array([0,0,0])
        self.tau_mx = tau_max
        self.J = 1
        self.dt = dt

    def step(self, jnt_err):
        dedt = (jnt_err - self.prev_jnt_err)/self.dt
        self.jnt_err = jnt_err
        tau = self.P*self.jnt_err + self.D*dedt
        tau = np.clip(tau, -self.tau_mx, self.tau_mx)
        return tau, dedt

class action_space():
    def __init__(self):
        self.shape = np.array([3]) # three joint angles adjustments
        self.high = np.ones(3) * jnt_vel_max
        self.low = np.ones(3) * -jnt_vel_max

    # def sample(self):
    #     return 2*torch.rand(3) - 0.5

class observation_space():
    def __init__(self):
        self.shape = np.array([3])  # [x, y, dxdt, dydt, th1, th2, th3, w1, w2, w3]
                                    # [th1, th2, th3, w1, w2, w3]

class RobotEnv():
    # have to generate random poses
    def __init__(self, eval=False, has_objects=True):
        self.robot = Robot.robot_3link()
        if has_objects:
            self.objs = [rand_object(dt=dt), rand_object(dt=dt), rand_object(dt=dt)]
        else:
            self.objs = [] 

        self.action_space = action_space()
        self.observation_space = observation_space()
        self.reward_range = [0, -np.inf]
        self.Controller = PDControl(dt=dt)
        self.eval = eval

        # setting runtime variables 
        s = gen_rand_pos()
        g = gen_rand_pos()
        th_arr = self.robot.reverse(goal=s)
        self.start = th_arr[:,0]
        th_arr = self.robot.reverse(goal=g)
        self.goal = th_arr[:,0]
        self.robot.set_pose(self.start)
        self.done = False
        self.jerk_sum = 0
        self.t_sum = 0
        self.t_count = 0
        # self.jnt_err_sum = 0
        self.info = {}
        self.jnt_err = calc_jnt_err(self.robot.pos, self.goal) 
        self.jnt_err_vel = np.array([0,0,0])
        self.prev_tau = np.array([0,0,0])


    # need to return the relative positions of the object and the relative vels
    # in terms of the end effector frame of reference.
    def step(self, action, use_PID=False):
        self.t_count += 1
        paused = False
        for o in self.objs:
            prox = self.robot.proximity(o)
            if np.any(prox <= min_prox):
                paused = True

        if not paused:
            if use_PID:
                tau, dedt = self.Controller.step(self.jnt_err)
            else:
                if not isinstance(action,np.ndarray):
                    action = action.cpu()
                    action = action.detach()
                    action = action.numpy()
                action = action.reshape(3)
                tau = action

            nxt_vel = (tau-damping*self.robot.jnt_vel)*dt + self.robot.jnt_vel
            self.robot.set_jnt_vel(nxt_vel) 
            nxt_pos = angle_calc(dt * self.robot.jnt_vel + self.robot.pos)
            self.robot.set_pose(nxt_pos) # set next pose
            jnt_err = calc_jnt_err(self.robot.pos, self.goal)

            self.jnt_err_vel = (self.jnt_err - jnt_err)/dt
            self.jnt_err = jnt_err # joint error for stateself.n_actions
            # self.jnt_err_sum += np.linalg.norm(self.jnt_err*dt)
        else:
            self.robot.set_jnt_vel(np.array([0,0,0]))
            jnt_err = calc_jnt_err(self.robot.pos, self.goal)
            self.jnt_err_vel = (self.jnt_err - jnt_err)/dt
            self.jnt_err = jnt_err

        # update objects
        for o in self.objs:
            o.step()

        bonus = 0
        self.t_sum += dt
        if self.t_sum > t_limit:
            done = True
        elif np.all(abs(self.jnt_err) < thres) and np.all(abs(self.jnt_err_vel) < vel_thres): 
            done = True 
            bonus = 100
            print('finished by converging!!')
        else:
            done = False

        reward = -Alpha * np.linalg.norm(self.jnt_err) + bonus

        coords = []
        feats = []
        if not self.eval: # skip over computation when making an animation
            rob_coords, rob_feats = self.robot.get_coords(self.t_count)
            coords.append(rob_coords)
            feats.append(rob_feats)
            for obj in self.objs:
                c,f = obj.get_coords(self.t_count)
                coords.append(c)
                feats.append(f)
            coords = np.vstack(coords)
            feats = np.vstack(feats)
        state = (coords,feats,self.jnt_err)
        return state, reward, done, self.info


    def reset(self):
        self.__init__()
    
        # state = np.hstack((jnt_err, jnt_err_vel, rel_pos, rel_vel))
        # state = np.hstack((self.jnt_err, self.jnt_err_vel))
        coords = []
        feats = []
        for obj in self.objs:
            c,f = obj.get_coords(self.t_count)
            coords.append(c)
            feats.append(f)
        rob_coords, rob_feats = self.robot.get_coords(t=self.t_count)
        coords.append(rob_coords)
        feats.append(rob_feats)
        coords = np.vstack(coords)
        feats = np.vstack(feats)

        state = (coords,feats,self.jnt_err)
        return state


    


        



