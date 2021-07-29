from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns
# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, dataloader, sampler
import torch.nn as nn
from torch.optim import lr_scheduler
import wandb
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd
import os

# Image manipulations
from PIL import Image
# Useful for examining network
from torchsummary import summary
# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14

#custom class
from dataset import MyDataset
from slowfastnet import SlowFast,Bottleneck
from TrainTestCode import train,mapIntToClass,check_accuracy,getTop_k


run = wandb.init(project="kids_model",config='config-default.yaml')
config = wandb.config
age = config['age']
# Printing out all outputs
InteractiveShell.ast_node_interactivity = 'all'

# %% [markdown]
# # SET UP TARAINING PARAMS (GPU OR CPU)

# %%
batch_size = 32
save_models = f"model_save/checkpoint_{wandb.run.name}_{age}.pt"
# checkpoint_path = "Kid-specificModel.pth" #saving model
n_classes = 21
# Whether to train on a gpu

device = 'cuda'
print(f'Device: {device}')


# Number of gpus
# %% [markdown]
# # Create Data Loader for training and test split

"""
This part can be replaced depending on how your store your train-val-test-split.
We set the age paramater in config file to account for kid/adult training file as explained in the paper
MyDataset is the custom file that handles the reading of the input videos as well as selecting the number of frames needed.
We use wandb.ai to handle the visualization of our runs. Detailed tutorial on how to use it can be found on there website.
"""
train_path = r'data\Data_Csv\train_mixed2.csv'
val_path = f'data/Data_Csv/val_mixed.csv'
test_path  = f'data/Data_Csv/test_mixed.csv'
dataset_train = MyDataset(train_path,mode='train',target_n_frames=config['target_n_frames'])
dataset_val = MyDataset(val_path,mode='val',target_n_frames=config['target_n_frames'])
dataset_test = MyDataset(test_path,mode='test',target_n_frames=config['target_n_frames'])
dataLoader = {
    'train':DataLoader(dataset_train,batch_size= batch_size,shuffle=True),
    'test': DataLoader(dataset_test,batch_size= batch_size,shuffle=True),
    'val':DataLoader(dataset_val,batch_size= batch_size,shuffle=True)
}

artifact = wandb.Artifact('my-dataset', type='dataset')
artifact.add_file(train_path)
artifact.add_file(val_path)
artifact.add_file(test_path)
run.log_artifact(artifact)


# %% [markdown]
# # Load Model and freeze previous layers

# %%
model = SlowFast(Bottleneck, [3, 4, 6, 3],num_classes=200)
num_ftrs = model.fc.in_features #we initialize the model with previous weights and only finetune the last last year
if not config['checkpoint_path']:
    print(f'Loading original pretrained weights with new head')
    state_dict = torch.load('pretrained_models/slowfast50_best_fixed.pth',map_location=device)#remove map_location on Gpu
    model.load_state_dict(state_dict)
    model.fc =  nn.Linear(num_ftrs, 21)
    model.epoch = 0
else:
    print("Loading from checkpoint: " + config['checkpoint_path'])
    state_dict = torch.load(config['checkpoint_path'],map_location=device)#remove map_location on Gpu
    model.fc =  nn.Linear(num_ftrs, 21)
    model.load_state_dict(state_dict)
    model.epoch = 0
    
for param in model.parameters():
    param.requires_grad = False #freeze all layers
for param in model.fc.parameters(): # Unfreeze head
    param.requires_grad = True
    

model = model.to(device)


# %% [markdown]
# # Configure training loss and optimizer

# %%
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model.fc.parameters(), lr=config['lr'])
# Decay LR if plateau
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=config['patience'], factor=0.1,threshold=0.01)

# %% [markdown]
# # Train Model




# %%
model, history = train(config,
    model,
    criterion,
    optimizer_ft,
    dataLoader['train'],
    dataLoader['val'],
    exp_lr_scheduler,
    save_models,device = device,max_epochs_stop=config['max_epochs_stop'],n_epochs = config['epochs'])


torch.save(model.state_dict(),f'model_save/{wandb.run.name}_{age}.pt' )
