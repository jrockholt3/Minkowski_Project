import numpy as np
import torch
from Agent import Agent
from Robot_Env import RobotEnv
import GPUtil
import gc 
from time import time 

def check_memory():
    q = 0
    for obj in gc.get_objects():
        try:  
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                q += 1
        except:
            pass
    return q 

load_check_ptn = False
episodes = 100
best_score = -np.inf
n = 5 # number of episodes to calculate the average score
n_batch = 10 # number of batches to train the networks over per episode

env = RobotEnv()
agent = Agent(env,batch_size=128)

if load_check_ptn:
    agent.load_models()

score_history = []
for i in range(episodes):
    with torch.no_grad():
        state = env.reset()
        n = 0
        done = False
        score = 0
        t1 = time()
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, new_state,done,n)
            score += reward
            state = new_state

        score_history.append(score)
        if np.mean(score_history[-n:]) > best_score:
            print('saving models')
            agent.save_models()
            best_score = np.mean(score_history[-n:])

    print('episode', i, 'score_avg %.2f' %np.mean(score_history[-n:]), 'time %.2f' %(time()-t1))

    for j in range(n_batch):
        agent.learn()

    torch.cuda.memory_summary(abbreviated=False)
    torch.cuda.empty_cache()
    gc.collect()
    print('episode',i,'added',check_memory())
    # print('memory allocated 0: %f' %(torch.cuda.memory_allocated(0)))
    # GPUtil.showUtilization()

