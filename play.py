'''
MIT License
Copyright (c) 2022 HuangChenrui<hcr2077@outlook.com>
'''

# Highway Env
import gym
import highway_env

# Other Lib
import numpy as np
from DQN import *
from DoubleDQN import *
from DuelDQN import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# config
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted",
    },
    "duration": 100,
    "collision_reward": -3, # needs to change 
}

if __name__ == '__main__':
    env = gym.make('highway-fast-v0')
    env.configure(config)
    dqn = DuelDQN(is_train=False)
    dqn.load("./91check_points.tar")
    print('--------------\nLoading experience...\n--------------')

    total_reward = 0
    for i_episode in range(100000):
        s = env.reset()
        s = np.squeeze(np.reshape(s,(1,dqn.N_STATES)))

        while True:
            env.render()
            # take action based on the current state
            action_index, action_value = dqn.choose_action(s)
            # choose action
            action = action_index

            # print('\r' + "action: " + str(action), end='', flush=True)
            # obtain the reward and next state and some other information
            s_, reward, done, info = env.step(action)
            # slice s and s_
            s = np.squeeze(np.reshape(s,(1,dqn.N_STATES)))
            s_ = np.squeeze(np.reshape(s_,(1,dqn.N_STATES)))
            total_reward += reward
            s = s_  
            if done :
                print( 'Ep: ', i_episode, ' |', 'Ave_r:', round(total_reward/(1+i_episode),2) )
                break

