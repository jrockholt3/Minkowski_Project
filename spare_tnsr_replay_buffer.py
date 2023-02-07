import numpy as np
import pickle 
import torch

class ReplayBuffer:
    def __init__(self, max_size, jnt_d, time_d, file='replay_buffer'):
        # jnt_d = joint dimensions
        self.mem_size = max_size
        self.mem_cntr = 0
        self.time_d = time_d # this is the # of time steps that will be looked over
        # coord_memory, feat_memory, and jnt_err all define the state 
        self.coord_memory = np.empty(max_size,dtype=np.object)
        self.feat_memory = np.empty(max_size,dtype=np.object)
        self.jnt_err_memory = torch.zeros((self.mem_size, jnt_d))

        self.new_coord_memory = np.empty(max_size,dtype=np.object)
        self.new_feat_memory = np.empty(max_size,dtype=np.object)
        self.new_jnt_err_memory = torch.zeros((self.mem_size, jnt_d))

        self.action_memory = torch.zeros((self.mem_size, jnt_d))
        self.reward_memory = torch.zeros(self.mem_size)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.bool)
        self.file = file
        self.time_step = torch.ones(self.mem_size)*np.inf # variable for sampling buffer
                                        # will need to go back 'x' # of time steps
                                        # to create the 4D space tensor
                                        # set to np.inf so that all entries do not 
                                        # have a state stored will have a greater timestep
                                        # and avoid storing empty arrays

    def store_transition(self, state, action, reward, new_state, done):
        # state is (coord_list, feat_list, jnt_err, time_step)
        ndx = self.mem_cntr % self.mem_size

        self.coord_memory[ndx] = state[0]
        self.feat_memory[ndx] = state[1]
        self.jnt_err_memory[ndx] = state[2]
        self.time_step[ndx] = state[3]
        self.new_coord_memory[ndx] = new_state[0]
        self.new_feat_memory[ndx] = new_state[1]
        self.new_jnt_err_memory[ndx] = new_state[2]
        self.terminal_memory[ndx] = done
        self.action_memory[ndx] = action
        self.reward_memory[ndx] = reward
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        min_mem = min(self.mem_cntr, self.mem_size)
        print('mem_cntr =', self.mem_cntr)


        batch = np.random.choice(min_mem, batch_size, replace=False)
        
        coord_batch = []
        feat_batch = []
        new_coord_batch = []
        new_feat_batch = []
        coord_list = [] 
        new_coord_list = []
        feat_list = []
        new_feat_list = []
        for b in batch:
            for t in range(self.time_d):
                ndx_i = (b - t) % self.mem_size
                if ndx_i >= 0: # check if ndx is out of range
                    if self.time_step[ndx_i]<=self.time_step[b]: # check if still in same episode
                        coord_list.append(self.coord_memory[ndx_i])
                        new_coord_list.append(self.new_coord_memory[ndx_i])
                        feat_list.append(self.feat_memory[ndx_i])
                        new_feat_list.append(self.new_feat_memory[ndx_i])
                    else:
                        t = self.time_d + 1
                else:
                    t = self.time_d + 1

            coord_batch.append(torch.vstack(coord_list))
            feat_batch.append(torch.vstack(feat_list))
            new_coord_batch.append(torch.vstack(new_coord_list))
            new_feat_batch.append(torch.vstack(new_feat_list))
            feat_list = []
            coord_list = []
            new_coord_list = []
            new_feat_list = []

        jnt_err = self.jnt_err_memory[batch]
        new_jnt_err = self.new_jnt_err_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return (torch.vstack(coord_batch), torch.vstack(feat_batch), jnt_err), actions, \
                rewards, (torch.vstack(new_coord_batch), torch.vstack(new_feat_batch), new_jnt_err), dones
        




    def save(self):
        print('saving buffer')
        file_str = self.file + '.pkl'
        with open(file_str, 'wb') as file:
            pickle.dump(self,file)
        print('buffer saved')

    def load(self):
        print('loading buffer')
        file_str = self.file + '.pkl'
        with open(file_str, 'rb') as file:
            new_buff = pickle.load(file)

        # coord_memory, feat_memory, and jnt_err all define the state 
        self.coord_memory = new_buff.coord_memory
        self.feat_memory = new_buff.feat_memory
        self.jnt_err_memory = new_buff.jnt_err_memory

        self.new_coord_memory = new_buff.new_coord_memory
        self.new_feat_memory = new_buff.new_feat_memory
        self.new_jnt_err_memory = new_buff.new_jnt_err_memory

        self.action_memory = new_buff.action_memory
        self.reward_memory = new_buff.reward_memory
        self.terminal_memory = new_buff.terminal_memory
        self.time_step = new_buff.time_step 