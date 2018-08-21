"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
    
        if (np.random.random() < (1 - self.epsilon)):
            #choose random action
            return np.random.choice(self.actions)
        else:
            #choose one with highest q value
            cur_action = self.q_table.loc[observation,:]
            cur_action = cur_action.reindex(np.random.permutation(cur_action.index))
            # print('cur action: ', cur_action)
            return np.argmax(cur_action)
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        old_reward = self.q_table.loc[s, a]
        if s_ != 'terminal':
            new_reward = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            new_reward = r # next state is terminal
        self.q_table.loc[s, a] += self.lr * (new_reward - old_reward) 

    def check_state_exist(self, state):
        if (state not in self.q_table.index):
            print('actions: ', self.actions)
            print('columns: ', self.q_table.columns)
            to_add = pd.Series(data=[0] * len(self.actions), index=self.q_table.columns, name=state)
            # print('to_add: ', to_add)
            self.q_table = self.q_table.append(
               to_add
            )
