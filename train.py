import numpy as np
import torch
from Agent import Agent
from Robot_Env import RobotEnv

load_check_ptn = False
episodes = 10
best_score = -np.inf
n = 5 # number of episodes to calculate the average score
n_batch = 10 # number of batches to train the networks over per episode

env = RobotEnv()
agent = Agent(env)

if load_check_ptn:
    agent.load_models()

score_history = []
for i in range(episodes):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(state)
        new_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, new_state,done)
        score += reward
        state = new_state

    score_history.append(score)
    if np.mean(score_history[-n:]) > best_score:
        agent.save_models()

    for i in range(n_batch):
        agent.learn()

    print('episode', i, 'score %.2f' %score, 'score_avg %.2f' %np.mean(score_history[-n:]))