'''
MIT License
Copyright (c) 2022 HuangChenrui<hcr2077@outlook.com>

Double DQN don't need to change DQN_Net,
but only need to change the policy to choose action.
'''

# Net
from DoubleDQN_Net import *

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Tensorboard
from torch.utils.tensorboard import SummaryWriter   

# Numpy
import numpy as np

# Hyper Parameters
BATCH_SIZE = 64
LR = 0.001                   # learning rate
EPSILON = 0.1               # greedy policy
SETTING_TIMES = 500         # greedy setting times 
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 1000   # target update frequency
MEMORY_CAPACITY = 50000

class DoubleDQN(object):
    def __init__(self, is_train=True):
        self.IS_TRAIN = is_train
        self.eval_net, self.target_net = DoubleDQN_Net(), DoubleDQN_Net()
        self.learn_step_counter = 0     # for target updating
        self.memory_counter = 0         # for storing memory
        # (s,a,r,s_)一共 2 * state + 2 个列,因为动作是唯一确定的
        self.memory = np.zeros((MEMORY_CAPACITY, self.eval_net.N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./log') if self.IS_TRAIN else None

        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.EPSILON = EPSILON if self.IS_TRAIN else 1.0
        self.SETTING_TIMES = SETTING_TIMES
        self.GAMMA = GAMMA
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.N_STATES = self.eval_net.N_STATES

    # 此时只选择action的序号，具体的action和epsilon的更改放在主函数中确定
    # epsilon greedy
    def choose_action(self, x):

        x = torch.unsqueeze(torch.FloatTensor(x), 0) # add 1 dimension to input state x

        if np.random.uniform() < self.EPSILON: # greedy
            action_value = self.eval_net.forward(x)
            # torch.max() 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
            action_index = torch.max(action_value, 1)[1].data.numpy()[0] # 此时已经转变为index的形式
            action_max_value = torch.max(action_value, 1)[0].data.numpy()[0]
        else:
            action_index = np.random.randint(0, self.eval_net.N_ACTIONS)
            action_max_value = 0
        return action_index, action_max_value

    # store memory
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # update the target network every fixed steps
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.eval_net.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.eval_net.N_STATES:self.eval_net.N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.eval_net.N_STATES+1:self.eval_net.N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.eval_net.N_STATES:])

        # diff between DQN
        q_eval = self.eval_net(b_s).gather(1, b_a)  # dim=1是横向的意思 shape (batch, 1)
        q_eval_max_a = self.eval_net(b_s_).detach()     # detach from graph, don't backpropagate
        # q_eval_max_a.max(1) returns the max value along the axis=1 and its corresponding index
        eval_max_a_index = q_eval_max_a.max(1)[1].view(BATCH_SIZE, 1)

        q_next = self.target_net(b_s_).gather(1, eval_max_a_index)

        q_target = b_r + GAMMA * q_next   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if (self.IS_TRAIN):
            if (self.learn_step_counter % 100000 == 0):
                self.writer.add_scalar('Loss', loss, self.learn_step_counter)

    def save(self,path):
        torch.save(self.eval_net.state_dict(), path)
    def load(self,path):
        self.eval_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))


