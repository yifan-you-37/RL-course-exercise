
import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name='s')
            self.a = tf.placeholder(dtype=tf.int32, shape=[None, ], name='a')
            self.discounted_reward = tf.placeholder(dtype=tf.float32, shape=[None,], name='discounted_reward')
        
        w_initializer = tf.initializers.random_normal(0, 0.3)
        b_initializer = tf.initializers.constant(0.1)
        with tf.variable_scope('net'):
            tmp = tf.layers.dense(self.s, 10, tf.nn.tanh, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='l1')        
            self.all_act_prob = tf.layers.dense(tmp, self.n_actions, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='l2')
        
        with tf.variable_scope('softmax'):
            self.output_prob = tf.nn.softmax(self.all_act_prob, name='softmax')
        
        with tf.variable_scope('loss'):
            unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.a, logits=self.output_prob, name='cross_entropy_loss_unweighted')
            self.loss = tf.reduce_mean(unweighted_loss * self.discounted_reward)
        
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train = optimizer.minimize(self.loss)


    def choose_action(self, observation):

        actions_prob = self.sess.run(self.output_prob, feed_dict={
            self.s: observation[np.newaxis, :]
        })
        return np.random.choice(np.arange(0,self.n_actions,1), p=actions_prob.ravel())

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        r = self._discount_and_norm_rewards()
        _, loss = self.sess.run([self.train, self.loss], feed_dict={
            self.s: self.ep_obs,
            self.a: self.ep_as,
            self.discounted_reward: r
        })
        self.ep_obs = []
        self.ep_as = []
        self.ep_rs = []
        return r

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



