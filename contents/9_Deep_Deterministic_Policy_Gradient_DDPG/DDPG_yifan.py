"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
OUTPUT_GRAPH = True
RENDER = True
ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        
        self.s = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s')
        self.s_ = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s_')
        self.r = tf.placeholder(tf.float32, [None,], name='r')
        
        self.a = self._build_a(self.s, 'actor_eval', True)
        self.a_ = self._build_a(self.s_, 'actor_target', False)
        self.q = self._build_c(self.s, self.a, 'critic_eval', True)
        self.q_ = self._build_c(self.s_, self.a_, 'critic_target', False)
        
        self.a_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_eval')
        self.a_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')
        self.c_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_eval')
        self.c_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_target')
        
        with tf.variable_scope('soft_replacement'):
            self.soft_replace_actor = [tf.assign(t, t * (1-TAU) + e * TAU) for t, e in zip(self.a_t_params, self.a_e_params)]
            self.soft_replace_critic = [tf.assign(t, t * (1 - TAU) + e * TAU) for t, e in zip(self.c_t_params, self.c_e_params)]
            
        with tf.variable_scope('train_actor'):
            loss_actor = -tf.reduce_mean(self.q)
            self.train_actor = tf.train.AdamOptimizer(LR_A).minimize(loss_actor, var_list=self.a_e_params)
        
        with tf.variable_scope('train_critic'):
            q_target = self.r + GAMMA * self.q_
            loss_critic = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
            self.train_critic = tf.train.AdamOptimizer(LR_C).minimize(loss_critic, var_list=self.c_e_params)
        
        
        
        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.s: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run([self.soft_replace_actor, self.soft_replace_critic])

        batch_index = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)

        batch = self.memory[batch_index, :]
        s = batch[:, :self.s_dim]
        a = batch[:,  self.s_dim:self.s_dim + self.a_dim]
        r = batch[:, -self.s_dim - 1: -self.s_dim].ravel()
        s_ = batch[:, -self.s_dim:]

        _ = self.sess.run([self.train_actor], {
            self.s: s,
        })

        _ = self.sess.run([self.train_critic], {
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_
        })

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):

        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 30, tf.nn.relu, trainable=trainable, name='l1')
            l2 = tf.layers.dense(l1, self.a_dim, tf.nn.tanh, trainable=trainable)
            return tf.multiply(l2, self.a_bound, name='scale_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            feed = tf.concat([s, a], 1, name='concat')
            l1 = tf.layers.dense(feed, 30, tf.nn.relu, trainable=trainable, name='l1')
            l2 = tf.layers.dense(l1, 1, trainable=trainable, name='l2')
            return l2


###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER and i>100:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)