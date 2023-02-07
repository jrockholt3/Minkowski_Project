import torch
import torch.nn as nn
from torch.optim import NAdam
import numpy as np
import MinkowskiEngine as ME
from spare_tnsr_replay_buffer import ReplayBuffer
from Networks import Actor, Critic
import Robot_Env
import pickle

class Agent():
    def __init__(self, input_dims, alpha=0.001,beat=0.002,env=None, 
                    gamma=.99, n_actions=3, time_d=6, max_size=int(1e6), tau=0.005,
                    batch_size=128,noise=.01,e=.1,enoise=.3):
        self.gamma = gamma
        self.memory = ReplayBuffer(max_size,n_actions,time_d)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.e = e
        self.enoise = enoise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.tau = tau
        self.score_avg = 0
        self.best_score = 0

        self.actor = Actor(1,n_actions,4)
        self.critic = Critic(1,4)
        self.target_actor = Actor(1,n_actions,4)
        self.target_critic = Critic(1,4)

        self.update_network_params(tau=1)

    def update_network_params(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_dict = dict(critic_params)
        actor_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_dict:
            critic_dict[name] = tau*critic_dict[name].clone() + \
                                (1-tau)*target_critic_dict[name].clone()
        self.target_critic.load_state_dict(critic_dict)

        for name in actor_dict:
            actor_dict[name] = tau*actor_dict[name].clone() + \
                                (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_dict)

    def get_action(self, net_input, jnt_err, evaluate=False):
        self.actor.eval()
        actions = self.actor(net_input, jnt_err)
        if not evaluate:
            e = np.random.random()
            if e <= self.e:
                noise = self.enoise
            else:
                noise = self.noise
            mean = torch.zeros_like(actions)
            actions += torch.normal(mean,noise)
        self.actor.train()
        actions = torch.clip(actions,min=self.min_action,max=self.max_action)
        return actions

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state,action,reward,new_state,done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        