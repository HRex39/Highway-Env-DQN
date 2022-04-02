'''
MIT License
Copyright (c) 2022 HuangChenrui<hcr2077@outlook.com>

Double DQN don't need to change DQN_Net,
but only need to change the policy to choose action.
'''

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Net Params
N_ACTIONS = 5 # Action Space is an array[steering, acceleration] like [-0.5,0.5] which need to be discrete
N_STATES = 25 # observation dim
N_LAYERS = [256, 256]

class DoubleDQN_Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.fc1 = nn.Linear(N_STATES, N_LAYERS[0])
        self.fc1.weight.data.normal_(0, 0.1) # initialization

        self.fc2 = nn.Linear(N_LAYERS[0], N_LAYERS[1])
        self.fc2.weight.data.normal_(0, 0.1) # initialization

        self.out = nn.Linear(N_LAYERS[1], N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1) # initialization

        self.N_ACTIONS = N_ACTIONS
        self.N_STATES = N_STATES

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # ReLU(Rectified Linear Unit,修正线性单元),取正
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
