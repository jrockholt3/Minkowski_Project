import numpy as np
import torch

def quantize(arr, res=0.01, workspace_limits=torch.tensor([[-.6,.6],[-.6,.6],[0,.9]])):
    if not torch.is_tensor(arr):
        arr = torch.tensor(arr)
    # this helper function takes in a 3xN array of (x,y,z) coords and
    # outputs the ndx of the coord based on a array representing the whole workspace
    # with resolution: "res"
    range_ = torch.abs(workspace_limits[:,1] - workspace_limits[:,0])
    ndx_range = range_/res 
    tnsr = torch.round(ndx_range * (arr - workspace_limits[:,0]) / range_, decimals=2)
    return tnsr.float()