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

top_only = True
transfer = True
load_check_ptn = False
load_memory = False
has_objs = True
episodes = 300
best_score = -640.0
n = 10 # number of episodes to calculate the average score
n_batch = 5 # number of batches to train the networks over per episode
batch_size = 384 # batch size

env = RobotEnv(has_objects=has_objs)
agent = Agent(env,batch_size=batch_size,max_size=50000,
                e=0.0,enoise=0.01*tau_max,top_only=top_only,
                transfer=transfer)

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
            ,'final jnt_err', np.round(env.jnt_err,2), 'time %.2f' %(time()-t1), ' mem ', check_memory())

file = open('tmp/loss_hist.pkl', 'wb')
pickle.dump(loss_hist,file)
file = open('tmp/score_history.pkl', 'wb')
pickle.dump(score_history, file)
agent.memory.save()
    # print('memory allocated 0: %f' %(torch.cuda.memory_allocated(0)))
    # GPUtil.showUtilization()

# fig = plt.figure()
# plt.plot(np.arange(len(score_history)), score_history)
# fig2 = plt.figure()
# plt.plot(np.arange(len(loss_hist)), loss_hist)
# plt.show()