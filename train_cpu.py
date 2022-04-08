'''
MIT License
Copyright (c) 2022 HuangChenrui<hcr2077@outlook.com>

Reference: 

'''

# Highway Env
import gym
import highway_env

# DQN\DDQN\DuelDQN
from DQN import *
from DoubleDQN import *
from DuelDQN import *
import numpy as np

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

'''
--------------Procedures of DQN Algorithm------------------
'''
if __name__ == '__main__':
    env = gym.make('highway-fast-v0')
    env.configure(config)
    dqn = DuelDQN(is_train=True)
    print('--------------\nCollecting experience...\n--------------')
    best_reward = 0

    for i_episode in range(100000):
        if i_episode <= dqn.SETTING_TIMES:
            dqn.EPSILON = 0.1 + i_episode / dqn.SETTING_TIMES * (0.9 - 0.1)
        s = env.reset()
        s = np.squeeze(np.reshape(s,(1,dqn.N_STATES)))
        # indirect params
        total_reward = 0
        total_action_value = 0
        action_counter = 1 
        reward_counter = 0
        bool_learning = False
        while True:
            # take action based on the current state
            action_index, action_value = dqn.choose_action(s)

            total_action_value += action_value
            if action_value != 0:
                action_counter += 1

            # choose action
            action = action_index
            # step
            s_, reward, done, info = env.step(action)
            # slice s and s_
            s = np.squeeze(np.reshape(s,(1,dqn.N_STATES)))
            s_ = np.squeeze(np.reshape(s_,(1,dqn.N_STATES)))
            # store the transitions of states
            dqn.store_transition(s, action_index, reward, s_)

            total_reward += reward
            reward_counter += 1
            
            # if the experience repaly buffer is filled, 
            # DQN begins to learn or update its parameters.    
            if dqn.memory_counter > dqn.MEMORY_CAPACITY:
                bool_learning = True
                dqn.learn()
            if done:
                # if game is over, then skip the while loop.
                if best_reward <= total_reward and bool_learning:
                    best_reward = total_reward
                    dqn.save("./"+str(round(best_reward))+"check_points.tar")
                if i_episode % 1000 == 999:
                    dqn.save("./"+str(i_episode)+".tar")
                print('Ep: ', i_episode, ' |', 'Ep_r: ', round(total_reward, 2), ' |', 'Best_r: ', round(best_reward, 2))
                break
            else:
                # use next state to update the current state. 
                s = s_
        if i_episode % 100 == 0:
            dqn.writer.add_scalar('Ep_r', total_reward, i_episode)
            dqn.writer.add_scalar('Ave_r', total_reward/reward_counter, i_episode)
            dqn.writer.add_scalar('Ave_Q_value', total_action_value/action_counter, i_episode)


