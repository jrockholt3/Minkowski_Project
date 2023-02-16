import numpy as np
import torch
from Agent import Agent
from Robot_Env import RobotEnv, tau_max
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

top_only=True
load_check_ptn = True
load_memory = False
has_objs = False
episodes = 100
best_score = -np.inf
n = 5 # number of episodes to calculate the average score
n_batch = 5 # number of batches to train the networks over per episode
batch_size = 512 # batch size

env = RobotEnv(has_objects=has_objs)
agent = Agent(env,batch_size=batch_size,max_size=5000,
                e=0.0,enoise=0.01*tau_max,top_only=top_only)

if load_check_ptn:
    agent.load_models()
if load_memory:
    agent.load_memory()

score_history = []
loss_hist = []
for i in range(episodes):
    t1 = time()
    with torch.no_grad():
        state = env.reset()
        n = 0
        done = False
        score = 0
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, new_state,done,n)
            score += reward
            state = new_state
            n += 1 

        score_history.append(score)
        if np.mean(score_history[-n:]) > best_score and i > 5:
            print('saving models')
            agent.save_models()
            best_score = np.mean(score_history[-n:])

    
    
    loss = 0
    if agent.memory.mem_cntr > batch_size:
        for j in range(n_batch):
            loss += agent.learn()
        loss_hist.append(loss/n_batch)
    print('episode', i, 'score %.2f' %score, 'score_avg %.2f' %np.mean(score_history[-n:]) \
            ,'final jnt_err', np.round(env.jnt_err,2), 'time %.2f' %(time()-t1))

file = open('tmp/loss_hist.pkl', 'wb')
pickle.dump(loss_hist,file)
file = open('tmp/score_history.pkl', 'wb')
pickle.dump(score_history, file)
agent.memory.save()
    # print('memory allocated 0: %f' %(torch.cuda.memory_allocated(0)))
    # GPUtil.showUtilization()

