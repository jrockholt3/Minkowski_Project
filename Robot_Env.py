import numpy as np
import Robot3D as Robot
from math import atan2
from Object import rand_object

# global variables
dt = 0.02 # time step
t_limit = 5 # time limit in seconds
thres = np.array([.03, .03, .03]) # joint error threshold
vel_thres = thres # joint velocity error threshold for stopping
# weighting different parts of the reward function
Alpha = 1 # area under joint err curve
Beta = 0 # positive reward for proximity
Gamma = 0 # negative rewards for be jumps in torque (minimize jerk)
prox_thres = .05 # proximity threshold - 5 cm
goal = np.array([np.pi/2, np.pi/2, np.pi/2])

# Controller gains
tau_max =  30 #J*rad/s^2, J = 1
damping = tau_max*.5
P = 10
D = 50

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
        self.high = np.ones(3) * tau_max
        self.low = np.ones(3) * -tau_max

    def sample(self):
        return 2*np.random.rand(3) - 0.5

class observation_space():
    def __init__(self):
        self.shape = np.array([3])  # [x, y, dxdt, dydt, th1, th2, th3, w1, w2, w3]
                                    # [th1, th2, th3, w1, w2, w3]

class RobotEnv():
    def __init__(self):
        self.robot = Robot.robot_3link()
        s = np.random.choice(4)
        g = np.random.choice(4)
        while s == g:
            g = np.random.choice(4)
        self.start = np.array([0,np.pi/4,-np.pi/2])
        self.goal = np.array([-np.pi/3, np.pi/12, -np.pi/3])
        self.robot.set_pose(self.start)
        self.shape = 10
        self.action_space = action_space()
        self.observation_space = observation_space()
        self.done = False
        self.jerk_sum = 0
        self.t_sum = 0
        self.jnt_err_sum = 0
        self.info = {}
        self.reward_range = [0, -np.inf]
        self.jnt_err = calc_jnt_err(self.robot.pos, self.goal) 
        self.jnt_err_vel = np.array([0,0,0])
        self.Controller = PDControl()
        self.prev_tau = np.array([0,0,0])

        self.obj = rand_object()



    # need to return the relative positions of the object and the relative vels
    # in terms of the end effector frame of reference.
    def step(self):
        tau, dedt = self.Controller.step(self.jnt_err)
        nxt_vel = (tau-damping*self.robot.jnt_vel)*dt + self.robot.jnt_vel
        self.robot.set_jnt_vel(nxt_vel) 
        nxt_pos = angle_calc(dt * self.robot.jnt_vel + self.robot.pos)
        self.robot.set_pose(nxt_pos) # set next pose
        jnt_err = calc_jnt_err(self.robot.pos, self.goal)

        self.jnt_err_vel = (self.jnt_err - jnt_err)/dt
        self.jnt_err = jnt_err # joint error for stateself.n_actions
        self.jnt_err_sum += np.linalg.norm(self.jnt_err*dt)

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

        # state = np.hstack((self.jnt_err, self.jnt_err_vel)) #, rel_pos, rel_vel))
        state = jnt_err
        # return state, reward, done, done, self.info
        return self.t_sum, self.robot.pos, done, 


    def reset(self):
        self.__init__()
    
        # state = np.hstack((jnt_err, jnt_err_vel, rel_pos, rel_vel))
        # state = np.hstack((self.jnt_err, self.jnt_err_vel))
        state = self.jnt_err
        return state


    


        



