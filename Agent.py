import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import NAdam
import numpy as np
import MinkowskiEngine as ME
from spare_tnsr_replay_buffer import ReplayBuffer
from Networks import Actor, Critic
import Robot_Env
import pickle

class Agent():
    def __init__(self, env, alpha=0.001,beat=0.002, 
                    gamma=.99, n_actions=3, time_d=6, max_size=int(1e6), tau=0.005,
                    batch_size=128,noise=.01,e=.1,enoise=.3):
        self.gamma = gamma
        self.memory = ReplayBuffer(max_size,n_actions,time_d)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.e = e
        self.enoise = enoise
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.tau = tau
        self.score_avg = 0
        self.best_score = 0

        self.actor = Actor(1,n_actions,4,name='actor')
        self.critic = Critic(1,1,4,name='critic')
        self.target_actor = Actor(1,n_actions,4,name='targ_actor')
        self.target_critic = Critic(1,1,4,name='targ_critic')

        self.update_network_params(tau=1) # hard copy

    def choose_action(self, state, evaluate=False):
        self.actor.eval()
        action = self.actor.forward(state,single_value=True)

        if not evaluate:
            e = np.random.random()
            if e <= self.e:
                noise = self.enoise
            else:
                noise = self.noise
            action += torch.normal(torch.zeros_like(action),std=noise)

        action = torch.clip(action,self.min_action, self.max_action)
        return action

    def update_network_params(self, tau=None):
        if tau is None:
            tau = self.tau

        '''
        code not needed 
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        '''
        critic_dict = self.critic.state_dict()
        actor_dict = self.actor.state_dict()
        target_critic_dict = self.target_critic.state_dict()
        target_actor_dict = self.target_actor.state_dict()

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

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target=[]
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])

        
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        target = torch.tensor(target).cuda()
        target = target.view(self.batch_size,1)

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_params()

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()


        