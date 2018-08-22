"""
This part of code is the Deep Q Network (DQN) brain.

view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: r1.2
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.collections_eval = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.collections_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

        with tf.variable_scope('update_frozen_parameters'):
            self.update_frozen_parameters = [tf.assign(target, ev) for target, ev in zip(self.collections_target, self.collections_eval)]

        self.cost_his = []


    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features])
        self.s_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features])
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.a = tf.placeholder(dtype=tf.int32, shape=[None, ])

        w_initializer = tf.initializers.random_uniform(0, 0.3)
        b_initializer = tf.initializers.constant(0.1)
        # ------------------ build evaluate_net ------------------
        with tf.variable_scope("eval_net"):
            l1 = tf.layers.dense(self.s, 20, activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='l1')
            self.q_eval = tf.layers.dense(l1, self.n_actions,  kernel_initializer=w_initializer, bias_initializer=b_initializer, name='l2')

        # ------------------ build target_net ------------------
        with tf.variable_scope("target_net"):
            t1 = tf.layers.dense(self.s, 20, activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='l1')
            self.q_next = tf.layers.dense(t1, self.n_actions,  kernel_initializer=w_initializer, bias_initializer=b_initializer, name='l2')
        
        with tf.variable_scope("q_target"):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1)
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval_reduced'):
            a_index = tf.stack([tf.range(tf.size(self.a)), self.a], axis=1)
            self.q_eval_reduced = tf.gather_nd(self.q_eval, a_index)
        
        with tf.variable_scope("loss"):
            self.loss = tf.losses.mean_squared_error(self.q_target, self.q_eval_reduced, reduction=tf.losses.Reduction.MEAN)
        
        with tf.variable_scope('train'):
            optimizer = tf.train.RMSPropOptimizer(self.lr)
            self.train = optimizer.minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        if(np.random.uniform() < self.epsilon):
            observation = observation[np.newaxis, :]
            q_cur = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            return np.argmax(q_cur)
        else:
            return np.random.randint(0, self.n_actions)

    def learn(self):
        if (self.learn_step_counter % self.replace_target_iter == 0):
            self.sess.run(self.update_frozen_parameters)
        random_samples = self.memory[np.random.randint(0, self.memory_counter, min(self.memory_counter,self.batch_size))]
        _, loss = self.sess.run([self.train, self.loss], feed_dict={
            self.s: random_samples[:,:self.n_features],
            self.a: random_samples[:, self.n_features],
            self.r: random_samples[:, self.n_features + 1],
            self.a: random_samples[:, -self.n_features:]
        })
        self.cost_his.append(loss)

        self.learn_step_counter += 1
        
        self.epsilon = min(self.epsilon_max, self.epsilon + self.epsilon_increment)


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

if __name__ == '__main__':
    DQN = DeepQNetwork(3,4, output_graph=True)