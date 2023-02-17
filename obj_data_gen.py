import numpy as np
import pickle 
import torch
import gc
from Object_v2 import rand_object

rng = np.random.default_rng()

class ObjBuffer:
    def __init__(self, max_size, time_d=6, file='obj_buffer',dir='buffer'):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.coord_memory = np.empty(max_size,dtype=np.object)
        self.feat_memory = np.empty(max_size,dtype=np.object)
        self.time_d = time_d

        self.targ_memory = np.zeros((max_size,12),dtype=np.float32)
        self.time_step = np.ones(max_size)*np.inf
        self.file = file
        self.dir = dir



    def store_state(self,state,targ,time_step):
        ndx = self.mem_cntr % self.mem_size
        self.coord_memory[ndx] = state[0]
        self.feat_memory[ndx] = state[1]
        self.time_step[ndx] = time_step
        self.targ_memory[ndx] = targ

        self.mem_cntr += 1

    def sample_buffer(self,batch_size):
        min_mem = min(self.mem_cntr, self.mem_size)
        batch = rng.choice(min_mem,batch_size,replace=False)

        coords_batch = []
        feat_batch = []
        for b in batch:
            feat_list = []
            coord_list = []
            for t in range(self.time_d):
                ndx_i = (b - t) % self.mem_size
                if ndx_i >= 0:
                    if self.time_step[ndx_i]==self.time_step[b]-t:
                        coord_list.append(self.coord_memory[ndx_i])
                        feat_list.append(self.feat_memory[ndx_i])
                    else:
                        assert('Could not finish batch')
                else:
                    assert('Could not finish batch')
            
            coords_batch.append(np.vstack(coord_list))
            feat_batch.append(np.vstack(feat_list))

        vel_arr = self.targ_memory[batch]
        state = (coords_batch, feat_batch)
        return state, vel_arr

    def save(self):
        print('saving buffer')
        file_str = self.dir + '/' + self.file + '.pkl'
        with open(file_str, 'wb') as file:
            pickle.dump(self, file)
    
    def load(self):
        print('loading buffer')
        file_str = self.dir + '/' + self.file + '.pkl'
        with open(file_str, 'rb') as file:
            new_buff = pickle.load(file)
        
        return new_buff


# episodes = int(np.round(2000/5))
# memory = ObjBuffer(int(1e6),file='val_obj_buffer')
# for i in range(episodes):
#     obj_pos = rand_object()
#     obj_pos.label = 1.0
#     obj_neg = rand_object()
#     obj_neg.label = -1.0
#     done = False
#     t = 0
#     while not done:
#         c1,f1 = obj_pos.get_coords(t)
#         c2,f2 = obj_neg.get_coords(t)
#         coords = np.vstack([c1,c2])
#         feats = np.vstack([f1,f2])
#         state = (coords,feats)
#         targ = np.hstack(((obj_pos.vel, obj_pos.curr_pos, obj_neg.vel, obj_neg.curr_pos)))
#         memory.store_state(state, targ, t)
#         obj_pos.step(t)
#         obj_neg.step(t)
#         t += 1
#         if np.round(np.linalg.norm(obj_pos.vel)) == 0 or np.round(np.linalg.norm(obj_neg.vel)) == 0:
#             done = True

# memory.save()
# print(memory.mem_cntr)
