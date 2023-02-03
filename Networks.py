import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np
from Object import rand_object, Cylinder

class Actor(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(Actor, self).__init__(D)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU()
        )
        self.pooling = ME.MinkowskiGlobalPooling()
        self.linear = ME.MinkowskiLinear(64, out_feat)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.linear(x)
        return x

obj = rand_object()
cyl = Cylinder()

out1,feat1 = obj.get_coord_list()
out2,feat2 = cyl.get_coord_list(0)

coords = np.vstack([out1,out2])
feats = np.vstack([feat1,feat2])
coords,feats = ME.utils.sparse_collate([coords],[feats])

input = ME.SparseTensor(feats,coords)
input.float()
print(input.dtype)
net = Actor(in_feat=3, out_feat=1, D=4)
print(net.forward(input))
print(input.shape)