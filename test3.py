import pickle
import spare_tnsr_replay_buffer
from Networks import Actor 

actor = Actor(.001,1,3,4,'actor')

trainabale_params = []
for name,p in actor.named_parameters():
    if "conv" not in name:
        trainabale_params.append(p)
