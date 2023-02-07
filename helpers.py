import numpy as np
import torch

def quantize(arr, res=0.01, workspace_limits=np.array([[-.6,.6],[-.6,.6],[0,.9]])):
    # this helper function takes in a 3xN array of (x,y,z) coords and
    # outputs the ndx of the coord based on a array representing the whole workspace
    # with resolution: "res"
    range_ = np.abs(workspace_limits[:,1] - workspace_limits[:,0])
    ndx_range = range_/res 
    ndx = np.round(ndx_range * (arr - workspace_limits[:,0]) / range_)
    tnsr = torch.from_numpy(ndx)
    return tnsr.float()