import numpy as np
import torch 
from obj_pred_network import Obj_Pred_Net
from torch.optim import NAdam
import torch.nn as nn
from obj_data_gen import ObjBuffer


# create the data base for the training

# make a data loader 

# for i in epochs - train 
epochs = 5
loss_hist = []
lr = .001
batch_size= 512

network = Obj_Pred_Net(lr,in_feat=1,D=4,name='obj_vel_pred')

optim = NAdam(params=network.parameters(),lr=lr)
criterion = nn.MSELoss()

train_data = ObjBuffer(int(1e6),file='obj_buffer')
train_data = train_data.load()
val_data = ObjBuffer(int(1e6),file='val_obj_buffer')
val_data = val_data.load()

n_batch = 0
max_batch = int(np.round(train_data.mem_cntr/batch_size))
best_score = -5
saved = False
for i in range(epochs):
    train_loss_sum = 0
    val_loss_sum = 0
    for j in range(max_batch):
        network.train()
        optim.zero_grad()
        state,y = train_data.sample_buffer(batch_size=batch_size)
        x,y = network.preprocessing(state,y)
        train_preds = network.forward(x)
        train_loss = criterion(train_preds.float(),y.float())
        train_loss.backward()
        optim.step()
        train_loss_sum += train_loss.item()

        network.eval()
        state_val,y_val = val_data.sample_buffer(batch_size=batch_size)
        x_val,y_val = network.preprocessing(state_val,y_val)
        val_preds = network.forward(x_val)
        val_loss = criterion(val_preds.float(),y_val.float())
        val_loss_sum += val_loss.item()

    if (val_loss_sum/max_batch) > best_score:
        saved = True
        best_score = val_loss_sum/max_batch
        network.save_checkpoint()
    
    print('epoch',i,'train_loss %.4f' %(train_loss_sum/max_batch), 'val_loss %.4f' %(val_loss_sum/max_batch))

if not saved:
    network.save_checkpoint()



# save model and weights 