'''
MIT License
Copyright (c) 2022 HuangChenrui<hcr2077@outlook.com>
'''

from torch.utils.tensorboard import SummaryWriter

from DuelDQN_Net import *

writer = SummaryWriter()
model = DuelDQN_Net()
input = torch.randn(1, model.N_STATES)
writer.add_graph(model, input)
writer.close()