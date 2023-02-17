import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import NAdam
import numpy as np
import MinkowskiEngine as ME
from spare_tnsr_replay_buffer import ReplayBuffer
from Networks import Actor, Critic
# import Robot_Env
import pickle
import gc 
from obj_pred_network import Obj_Pred_Net
from Robot_Env import tau_max

def check_memory():
    q = 0
    for obj in gc.get_objects():
        try:  
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                q += 1
        except:
            pass
    return q 

class Agent():
    def __init__(self, env, alpha=0.001,beta=0.002, gamma=.99, n_actions=3, 
                time_d=6, max_size=int(1e6), tau=0.005,
                batch_size=64,noise=.01*tau_max,e=.1,enoise=.1*tau_max,
                top_only=False,transfer=False):
        self.gamma = gamma
        self.memory = ReplayBuffer(max_size,n_actions,time_d)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.e = e
        self.enoise = enoise
        self.max_action = torch.tensor(env.action_space.high, device='cuda')
        self.min_action = torch.tensor(env.action_space.low, device='cuda')
        self.tau = tau
        self.score_avg = 0
        self.best_score = 0

        self.actor = Actor(alpha, 1,n_actions,4,name='actor',top_only=top_only)
        self.critic = Critic(beta, 1,4,name='critic',top_only=top_only)
        self.target_actor = Actor(alpha, 1,n_actions,4,
                                    name='targ_actor',top_only=top_only)
        self.target_critic = Critic(beta, 1,4,name='targ_critic',top_only=top_only)
        self.critic_criterion = nn.MSELoss()

        if transfer:
            temp = Obj_Pred_Net(lr=.001,in_feat=1,D=4)
            temp.load_checkpoint()
            self.actor.conv1.load_state_dict(temp.conv1.state_dict())
            self.actor.conv2.load_state_dict(temp.conv2.state_dict())
            self.actor.conv3.load_state_dict(temp.conv3.state_dict())
            self.actor.conv4.load_state_dict(temp.conv4.state_dict())
            self.critic.conv1.load_state_dict(temp.conv1.state_dict())
            self.critic.conv2.load_state_dict(temp.conv2.state_dict())
            self.critic.conv3.load_state_dict(temp.conv3.state_dict())
            self.critic.conv4.load_state_dict(temp.conv4.state_dict())
            del temp

        self.update_network_params(tau=1) # hard copy

    def choose_action(self, state, evaluate=False):
        self.actor.eval()
        # q = check_memory()
        x,jnt_err = self.actor.preprocessing(state,single_value=True)
        # print('preprocessing added',check_memory()-q)
        action = self.actor.forward(x,jnt_err)
        # print('forward pass added',check_memory()-q)

        if not evaluate:
            e = np.random.random()
            if e <= self.e:
                noise = self.enoise
            else:
                noise = self.noise
            action += torch.normal(torch.zeros_like(action),std=self.noise)

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

    def remember(self, state, action, reward, new_state, done,n):
        self.memory.store_transition(state,action,reward,new_state,done,n)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # target actions
        x,jnt_err = self.target_actor.preprocessing(state)
        target_actions = self.target_actor.forward(x,jnt_err)
        # target critic value
        x,jnt_err,target_actions = self.target_critic.preprocessing(new_state, target_actions)
        critic_value_ = self.target_critic.forward(x,jnt_err,target_actions)
        # critic value
        x,jnt_err,action = self.target_critic.preprocessing(new_state, target_actions)
        critic_value = self.target_critic.forward(x,jnt_err,action)

        target=[]
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = torch.vstack(target)
        
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = self.critic_criterion(target,critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        target = torch.tensor(target).cuda()
        target = target.view(self.batch_size,1)

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        # actions
        x,jnt_err = self.actor.preprocessing(state)
        mu = self.actor.forward(x,jnt_err)

        self.actor.train()
        # actor loss
        x,jnt_err,mu = self.critic.preprocessing(state,mu)
        actor_loss = -self.critic.forward(x,jnt_err, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_params()
        return critic_loss.item()

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

    def load_memory(self):
        self.memory = self.memory.load()


        