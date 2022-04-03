'''
MIT License
Copyright (c) 2022 HuangChenrui<hcr2077@outlook.com>

Duel DQN needs to change DQN_Net,
split it to A_Net and V_Net...
'''

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Net Params
N_ACTIONS = 5 # Action Space is an array[steering, acceleration] like [-0.5,0.5] which need to be discrete
N_STATES = 25 # observation dim
N_LAYERS = 256

class DuelDQN_Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.fc1 = nn.Linear(N_STATES, N_LAYERS)
        self.fc1.weight.data.normal_(0, 0.1) # initialization

        self.A_Net = nn.Linear(N_LAYERS, N_ACTIONS)
        self.A_Net.weight.data.normal_(0, 0.1) # initialization

        self.V_Net = nn.Linear(N_LAYERS, 1)
        self.V_Net.weight.data.normal_(0, 0.1) # initialization

        self.out = nn.Linear(N_ACTIONS, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1) # initialization

        self.N_ACTIONS = N_ACTIONS
        self.N_STATES = N_STATES

    def forward(self, x):
        # fc1
        x = self.fc1(x)
        x = F.relu(x) # ReLU(Rectified Linear Unit,修正线性单元),取正
        # A_Net
        x_A_Net = self.A_Net(x)
        # x_A_Net = F.relu(x_A_Net) # TODO:need RELU?
        # V_Net
        x_V_Net = self.V_Net(x)
        # x_V_Net = F.relu(x_V_Net)
        
        # Q = V(s) + A(s,a) - mean( A(s,a) )
        x_A_Net_Mean = torch.mean(x_A_Net, dim=1, keepdim=True)
        # expand x_V_Net(1*1) to x_V_Net(9*1), dim=0 -> column; dim=1 -> row
        x_V_Net = torch.cat([x_V_Net for i in range(N_ACTIONS)], dim=1)
        x_A_Net_Mean = torch.cat([x_A_Net_Mean for i in range(N_ACTIONS)], dim=1)
        
        x_Q_Value = x_V_Net + ( x_A_Net - x_A_Net_Mean )
        x_Q_Value = F.relu(x_Q_Value)

        actions_value = self.out(x_Q_Value)
        return actions_value








