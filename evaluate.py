import numpy as np
import matplotlib.pyplot as plt
import torch
from Agent import Agent
from Robot_Env import RobotEnv
import GPUtil
import gc 
from time import time 
import pickle

def check_memory():
    q = 0
    for obj in gc.get_objects():
        try:  
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                q += 1
        except:
            pass
    return q 

evaluate = True
load_check_ptn = True
load_memory = False
has_objs = False
best_score = -np.inf
n = 5 # number of episodes to calculate the average score
n_batch = 5 # number of batches to train the networks over per episode
batch_size = 512 # batch size

env = RobotEnv(has_objects=has_objs)
agent = Agent(env,batch_size=batch_size)

if load_check_ptn:
    agent.load_models()
if load_memory:
    agent.load_memory()

t1 = time()
obj_start_arr = []
obj_goal_arr = []
obj_vel_arr = []
for o in env.objs:
    obj_goal_arr.append(o.goal)
    obj_start_arr.append(o.start)
    obj_vel_arr.append(o.vel)
th_arr = []
jnt_err_arr = []
with torch.no_grad():
    state = env.reset()
    n = 0
    done = False
    score = 0
    th_arr.append(env.robot.get_pose())
    jnt_err_arr.append(env.jnt_err)
    while not done:
        action = agent.choose_action(state,evaluate=evaluate)
        new_state, reward, done, info = env.step(action)
        score += reward
        state = new_state
        th_arr.append(env.robot.get_pose())
        jnt_err_arr.append(env.jnt_err)
        n += 1 
    
    print('score %.2f' %score\
            ,'final jnt_err', np.round(env.jnt_err,2), 'time %.2f' %(time()-t1))

eval_dict = {'robot_pos':th_arr, 'obj_start':obj_start_arr, 'obj_goal':obj_goal_arr,'obj_vel':obj_vel_arr,
                'score':score, 'jnt_err_arr':jnt_err_arr}
file = open('tmp/eval_replay.pkl', 'wb')
pickle.dump(eval_dict,file)
    # print('memory allocated 0: %f' %(torch.cuda.memory_allocated(0)))
    # GPUtil.showUtilization()
fig = plt.figure()
plt.plot(np.arange(len(th_arr)), jnt_err_arr)
plt.show()