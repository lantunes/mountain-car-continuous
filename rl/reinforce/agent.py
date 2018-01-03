import numpy as np
import sklearn
import sklearn.pipeline
import tensorflow as tf
from sklearn.kernel_approximation import RBFSampler
from tensorflow.contrib import rnn


class TFRandomFeaturesStochasticPolicyAgent:
    def __init__(self, env, num_input=100, init_learning_rate=1e-4, min_learning_rate=1e-9, learning_rate_N_max=2000):
        # Feature Preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        self._env = env
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self._scaler = sklearn.preprocessing.StandardScaler()
        self._scaler.fit(observation_examples)

        # Used to convert a state to a featurized represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self._featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=25)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=25)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=25)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=25))
        ])
        self._featurizer.fit(self._scaler.transform(observation_examples))

        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")

        self._init_learning_rate = init_learning_rate
        self._min_learning_rate = min_learning_rate
        self._learning_rate_N_max = learning_rate_N_max
        self._learning_rate = tf.placeholder(tf.float32, shape=[])

        # policy parameters
        self._mu_theta = tf.get_variable("mu_theta", [1, 100],
                                         initializer=tf.zeros_initializer())
        self._sigma_theta = tf.get_variable("sigma_theta", [1, 100],
                                            initializer=tf.zeros_initializer())

        self._mu = tf.matmul(self._states, tf.transpose(self._mu_theta))
        self._sigma = tf.matmul(self._states, tf.transpose(self._sigma_theta))
        self._sigma = tf.nn.softplus(self._sigma) + 1e-5

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)

        self._discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 1), name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * self._sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * self._sigma**2)))

        self._gradients = self._optimizer.compute_gradients(self._loss)
        for i, (grad, var) in enumerate(self._gradients):
            if grad is not None:
                self._gradients[i] = (grad * self._discounted_rewards, var)
        self._train_op = self._optimizer.apply_gradients(self._gradients)

        self._sess.run(tf.global_variables_initializer())

        self._num_input = num_input
        # rollout buffer
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        # record reward history for normalization
        self._all_rewards = []
        self._max_reward_length = 1000000
        self._discount_factor = 0.99

    def _featurize_state(self, state):
        scaled = self._scaler.transform(np.array(state).reshape(1, len(state)))
        featurized = self._featurizer.transform(scaled)
        return featurized[0]

    def sample_action(self, system_state):
        # Gaussian policy
        system_state = self._featurize_state(system_state)
        mu, sigma = self._sess.run([self._mu, self._sigma], feed_dict={
            self._states: np.reshape(system_state, (1, self._num_input))
        })
        action = np.random.normal(mu, sigma)
        action = np.clip(action, self._env.action_space.low[0], self._env.action_space.high[0])
        return action, sigma

    def store_rollout(self, state, action, reward):
        self._action_buffer.append(action)
        self._reward_buffer.append(reward)
        state = self._featurize_state(state)
        self._state_buffer.append(state)

    def update_model(self, iteration):
        N = len(self._reward_buffer)
        r = 0 # use discounted reward to approximate Q value

        # compute discounted future rewards
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            # future discounted reward from now on
            r = self._reward_buffer[t] + self._discount_factor * r
            discounted_rewards[t] = r

        # reduce gradient variance by normalization
        self._all_rewards += discounted_rewards.tolist()
        self._all_rewards = self._all_rewards[:self._max_reward_length]
        discounted_rewards -= np.mean(self._all_rewards)
        discounted_rewards /= np.std(self._all_rewards)

        learning_rate = self._gen_learning_rate(iteration, l_max=self._init_learning_rate,
                                                l_min=self._min_learning_rate, N_max=self._learning_rate_N_max)

        for t in range(N-1):

            # prepare inputs
            states  = np.reshape(np.array(self._state_buffer[t]), (1, self._num_input))
            action = np.array(self._action_buffer[t])
            rewards = np.array([discounted_rewards[t]])

            # perform one update of training
            self._sess.run([self._train_op], {
                self._states:             states,
                self._taken_actions:      action,
                self._discounted_rewards: rewards,
                self._learning_rate:      learning_rate
            })
        self._clean_up()

    def _gen_learning_rate(self, iteration, l_max, l_min, N_max):
        if iteration > N_max:
            return l_min
        alpha = 2 * l_max
        beta = np.log((alpha / l_min - 1)) / N_max
        return alpha / (1 + np.exp(beta * iteration))

    def _clean_up(self):
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []


class TFNeuralNetStochasticPolicyAgent:
    def __init__(self, env, num_input, init_learning_rate=5e-6, min_learning_rate=1e-9, learning_rate_N_max=2000,
                 shuffle=True, batch_size=1, sigma=None):
        self._env = env
        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")

        self._init_learning_rate = init_learning_rate
        self._min_learning_rate = min_learning_rate
        self._learning_rate_N_max = learning_rate_N_max
        self._learning_rate = tf.placeholder(tf.float32, shape=[])

        # policy parameters
        self._mu_theta = tf.get_variable("mu_theta", [32, 1],
                                              initializer=tf.zeros_initializer())
        if sigma is None:
            self._sigma_theta = tf.get_variable("sigma_theta", [32],
                                                initializer=tf.zeros_initializer())

        # neural featurizer parameters
        self._W1 = tf.get_variable("W1", [num_input, 32],
                                   initializer=tf.random_normal_initializer())
        self._b1 = tf.get_variable("b1", [32],
                                   initializer=tf.constant_initializer(0))
        self._phi = tf.matmul(self._states, self._W1) + self._b1

        self._mu = tf.matmul(self._phi, self._mu_theta)
        if sigma is None:
            self._sigma = tf.reduce_sum(self._sigma_theta)
            self._sigma = tf.exp(self._sigma)
        else:
            self._sigma = tf.constant(sigma, dtype=tf.float32)

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)

        self._discounted_rewards = tf.placeholder(tf.float32, (None, 1), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 1), name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * self._sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * self._sigma**2))) * self._discounted_rewards

        self._train_op = self._optimizer.minimize(self._loss)

        self._sess.run(tf.global_variables_initializer())

        self._num_input = num_input
        self._shuffle = shuffle
        self._batch_size = batch_size
        # rollout buffer
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        # record reward history for normalization
        self._all_rewards = []
        self._max_reward_length = 1000000
        self._discount_factor = 0.99

        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self._scaler = sklearn.preprocessing.StandardScaler()
        self._scaler.fit(observation_examples)

    def sample_action(self, system_state):
        system_state = self._scaler.transform(system_state.reshape(1, -1))
        # Gaussian policy
        mu, sigma = self._sess.run([self._mu, self._sigma], feed_dict={
            self._states: np.reshape(system_state, (1, self._num_input))
        })
        action = np.random.normal(mu, sigma)
        action = np.clip(action, self._env.action_space.low[0], self._env.action_space.high[0])
        return action, sigma

    def store_rollout(self, state, action, reward):
        self._action_buffer.append(action)
        self._reward_buffer.append(reward)
        self._state_buffer.append(state)

    def update_model(self, iteration):
        N = len(self._reward_buffer)
        r = 0 # use discounted reward to approximate Q value

        # compute discounted future rewards
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            # future discounted reward from now on
            r = self._reward_buffer[t] + self._discount_factor * r
            discounted_rewards[t] = r

        # reduce gradient variance by normalization
        self._all_rewards += discounted_rewards.tolist()
        self._all_rewards = self._all_rewards[:self._max_reward_length]
        discounted_rewards -= np.mean(self._all_rewards)
        discounted_rewards /= np.std(self._all_rewards)

        learning_rate = self._gen_learning_rate(iteration, l_max=self._init_learning_rate,
                                                l_min=self._min_learning_rate, N_max=self._learning_rate_N_max)

        all_samples = []
        for t in range(N-1):
            state  = np.reshape(np.array(self._state_buffer[t]), self._num_input)
            action = self._action_buffer[t][0]
            reward = [discounted_rewards[t]]
            sample = [state, action, reward]
            all_samples.append(sample)
        if self._shuffle:
            np.random.shuffle(all_samples)

        batches = list(self._minibatches(all_samples, batch_size=self._batch_size))

        for b in range(len(batches)):
            batch = batches[b]
            states = [row[0] for row in batch]
            actions = [row[1] for row in batch]
            rewards = [row[2] for row in batch]

            self._sess.run([self._train_op], {
                self._states:             states,
                self._taken_actions:      actions,
                self._discounted_rewards: rewards,
                self._learning_rate:      learning_rate
            })

        self._clean_up()

    def _minibatches(self, samples, batch_size):
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def _gen_learning_rate(self, iteration, l_max, l_min, N_max):
        if iteration > N_max:
            return l_min
        alpha = 2 * l_max
        beta = np.log((alpha / l_min - 1)) / N_max
        return alpha / (1 + np.exp(beta * iteration))

    def _clean_up(self):
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []


class TFRecurrentStochasticPolicyAgent:
    def __init__(self, env, num_input, init_learning_rate=5e-6, min_learning_rate=1e-9, learning_rate_N_max=2000,
                 shuffle=True, batch_size=1):
        self._env = env
        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, 1, num_input), name="states")

        self._init_learning_rate = init_learning_rate
        self._min_learning_rate = min_learning_rate
        self._learning_rate_N_max = learning_rate_N_max
        self._learning_rate = tf.placeholder(tf.float32, shape=[])

        self._n_hidden = 32

        # policy parameters
        self._mu_theta = tf.get_variable("mu_theta", [self._n_hidden, 1],
                                         initializer=tf.zeros_initializer())
        self._sigma_theta = tf.get_variable("sigma_theta", [self._n_hidden],
                                            initializer=tf.zeros_initializer())

        # LSTM featurizer
        input_sequence = tf.unstack(self._states, 1, 1)
        self._lstm_cell = rnn.BasicLSTMCell(self._n_hidden, forget_bias=1.0)
        self._rnn_state_in = self._lstm_cell.zero_state(1, tf.float32)
        self._curr_rnn_state = (np.zeros([1, self._n_hidden]), np.zeros([1, self._n_hidden]))
        outputs, self._rnn_state = rnn.static_rnn(self._lstm_cell, input_sequence, dtype=tf.float32, initial_state=self._rnn_state_in)

        self._phi = outputs[-1]

        self._mu = tf.matmul(self._phi, self._mu_theta)
        self._sigma = tf.reduce_sum(self._sigma_theta)
        self._sigma = tf.exp(self._sigma)

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)

        self._discounted_rewards = tf.placeholder(tf.float32, (None, 1), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 1), name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * self._sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * self._sigma**2))) * self._discounted_rewards

        self._train_op = self._optimizer.minimize(self._loss)

        self._sess.run(tf.global_variables_initializer())

        self._num_input = num_input
        self._shuffle = shuffle
        self._batch_size = batch_size
        # rollout buffer
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        # record reward history for normalization
        self._all_rewards = []
        self._max_reward_length = 1000000
        self._discount_factor = 0.99

        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self._scaler = sklearn.preprocessing.StandardScaler()
        self._scaler.fit(observation_examples)

    def sample_action(self, system_state):
        system_state = self._scaler.transform(system_state.reshape(1, -1))
        # Gaussian policy
        mu, sigma, s = self._sess.run([self._mu, self._sigma, self._rnn_state], feed_dict={
            self._states: np.reshape(system_state, (1, 1, self._num_input)),
            self._rnn_state_in: self._curr_rnn_state
        })
        self._curr_rnn_state = s
        action = np.random.normal(mu, sigma)
        action = np.clip(action, self._env.action_space.low[0], self._env.action_space.high[0])
        return action, sigma

    def store_rollout(self, state, action, reward):
        self._action_buffer.append(action)
        self._reward_buffer.append(reward)
        self._state_buffer.append(state)

    def update_model(self, iteration):
        N = len(self._reward_buffer)
        r = 0 # use discounted reward to approximate Q value

        # compute discounted future rewards
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            # future discounted reward from now on
            r = self._reward_buffer[t] + self._discount_factor * r
            discounted_rewards[t] = r

        # reduce gradient variance by normalization
        self._all_rewards += discounted_rewards.tolist()
        self._all_rewards = self._all_rewards[:self._max_reward_length]
        discounted_rewards -= np.mean(self._all_rewards)
        discounted_rewards /= np.std(self._all_rewards)

        learning_rate = self._gen_learning_rate(iteration, l_max=self._init_learning_rate,
                                                l_min=self._min_learning_rate, N_max=self._learning_rate_N_max)

        all_samples = []
        for t in range(N-1):
            state  = [np.reshape(np.array(self._state_buffer[t]), self._num_input)]
            action = self._action_buffer[t][0]
            reward = [discounted_rewards[t]]
            sample = [state, action, reward]
            all_samples.append(sample)
        if self._shuffle:
            np.random.shuffle(all_samples)

        batches = list(self._minibatches(all_samples, batch_size=self._batch_size))

        curr_rnn_state = (np.zeros([1, self._n_hidden]), np.zeros([1, self._n_hidden]))

        for b in range(len(batches)):
            batch = batches[b]
            states = [row[0] for row in batch]
            actions = [row[1] for row in batch]
            rewards = [row[2] for row in batch]
            # perform one update of training
            _, s = self._sess.run([self._train_op, self._rnn_state], {
                self._states:             states,
                self._taken_actions:      actions,
                self._discounted_rewards: rewards,
                self._learning_rate:      learning_rate,
                self._rnn_state_in: curr_rnn_state
            })
            curr_rnn_state = s

        self._clean_up()

    def _minibatches(self, samples, batch_size):
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def _gen_learning_rate(self, iteration, l_max, l_min, N_max):
        if iteration > N_max:
            return l_min
        alpha = 2 * l_max
        beta = np.log((alpha / l_min - 1)) / N_max
        return alpha / (1 + np.exp(beta * iteration))

    def _clean_up(self):
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        self._curr_rnn_state = (np.zeros([1, self._n_hidden]), np.zeros([1, self._n_hidden]))


class TFAutoEncodingStochasticPolicyAgent:
    def __init__(self, env, num_input, init_learning_rate=5e-6, min_learning_rate=1e-9, learning_rate_N_max=2000,
                 shuffle=True, batch_size=1):
        self._env = env
        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")

        self._init_learning_rate = init_learning_rate
        self._min_learning_rate = min_learning_rate
        self._learning_rate_N_max = learning_rate_N_max
        self._learning_rate = tf.placeholder(tf.float32, shape=[])

        # policy parameters
        with tf.variable_scope("policy_params"):
            self._mu_theta = tf.get_variable("mu_theta", [32, 1],
                                             initializer=tf.zeros_initializer())
            self._sigma_theta = tf.get_variable("sigma_theta", [32],
                                                initializer=tf.zeros_initializer())

        # neural featurizer encoder
        with tf.variable_scope("autoencoder_params"):
            self._W1 = tf.get_variable("W1", [num_input, 32],
                                       initializer=tf.random_normal_initializer())
            self._b1 = tf.get_variable("b1", [32],
                                       initializer=tf.constant_initializer(0))
            self._phi = tf.matmul(self._states, self._W1) + self._b1

            # neural featurizer decoder
            self._W2 = tf.get_variable("W2", [32, 2],
                                       initializer=tf.random_normal_initializer(stddev=0.1))
            self._b2 = tf.get_variable("b2", [2],
                                       initializer=tf.constant_initializer(0))
            self._reconstruction = tf.nn.tanh(tf.matmul(self._phi, self._W2) + self._b2)

        self._mu = tf.matmul(self._phi, self._mu_theta)
        self._sigma = tf.reduce_sum(self._sigma_theta)
        self._sigma = tf.exp(self._sigma)

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)

        self._discounted_rewards = tf.placeholder(tf.float32, (None, 1), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 1), name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * self._sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * self._sigma**2))) * self._discounted_rewards

        policy_param_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy_params")
        self._train_op = self._optimizer.minimize(self._loss, var_list=policy_param_vars)

        # self._recon_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self._states, predictions=self._reconstruction))
        self._recon_loss = tf.sqrt(tf.reduce_mean(tf.square(self._states - self._reconstruction)))

        autoencoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "autoencoder_params")
        self._recon_train_op = self._optimizer.minimize(self._recon_loss, var_list=autoencoder_vars)

        self._sess.run(tf.global_variables_initializer())

        self._num_input = num_input
        self._shuffle = shuffle
        self._batch_size = batch_size
        # rollout buffer
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        # record reward history for normalization
        self._all_rewards = []
        self._max_reward_length = 1000000
        self._discount_factor = 0.99

        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self._scaler = sklearn.preprocessing.StandardScaler()
        self._scaler.fit(observation_examples)

    def sample_action(self, system_state):
        system_state = self._scaler.transform(system_state.reshape(1, -1))
        mu, sigma = self._sess.run([self._mu, self._sigma], feed_dict={
            self._states: np.reshape(system_state, (1, self._num_input))
        })
        action = np.random.normal(mu, sigma)
        action = np.clip(action, self._env.action_space.low[0], self._env.action_space.high[0])
        return action, sigma

    def store_rollout(self, state, action, reward):
        self._action_buffer.append(action)
        self._reward_buffer.append(reward)
        self._state_buffer.append(state)

    def update_model(self, iteration):
        N = len(self._reward_buffer)
        r = 0 # use discounted reward to approximate Q value

        # compute discounted future rewards
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            # future discounted reward from now on
            r = self._reward_buffer[t] + self._discount_factor * r
            discounted_rewards[t] = r

        # reduce gradient variance by normalization
        self._all_rewards += discounted_rewards.tolist()
        self._all_rewards = self._all_rewards[:self._max_reward_length]
        discounted_rewards -= np.mean(self._all_rewards)
        discounted_rewards /= np.std(self._all_rewards)

        learning_rate = self._gen_learning_rate(iteration, l_max=self._init_learning_rate,
                                                l_min=self._min_learning_rate, N_max=self._learning_rate_N_max)

        all_samples = []
        for t in range(N-1):
            state  = np.reshape(np.array(self._state_buffer[t]), self._num_input)
            action = self._action_buffer[t][0]
            reward = [discounted_rewards[t]]
            sample = [state, action, reward]
            all_samples.append(sample)
        if self._shuffle:
            np.random.shuffle(all_samples)

        batches = list(self._minibatches(all_samples, batch_size=self._batch_size))

        recon_losses = []
        for b in range(len(batches)):
            batch = batches[b]
            states = [row[0] for row in batch]
            _, recon = self._sess.run([self._recon_train_op, self._recon_loss], {
                self._states: states,
                self._learning_rate: learning_rate
            })
            recon_losses.append(recon)
        # print("mean recon loss: %s" % np.mean(recon_losses))

        for b in range(len(batches)):
            batch = batches[b]
            states = [row[0] for row in batch]
            actions = [row[1] for row in batch]
            rewards = [row[2] for row in batch]

            self._sess.run([self._train_op], {
                self._states:             states,
                self._taken_actions:      actions,
                self._discounted_rewards: rewards,
                self._learning_rate:      learning_rate
            })

        self._clean_up()

    def _minibatches(self, samples, batch_size):
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def _gen_learning_rate(self, iteration, l_max, l_min, N_max):
        if iteration > N_max:
            return l_min
        alpha = 2 * l_max
        beta = np.log((alpha / l_min - 1)) / N_max
        return alpha / (1 + np.exp(beta * iteration))

    def _clean_up(self):
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []


class TFDenoisingAutoEncodingStochasticPolicyAgent:
    def __init__(self, env, num_input, init_learning_rate=5e-6, min_learning_rate=1e-9, learning_rate_N_max=2000,
                 shuffle=True, batch_size=1):
        self._env = env
        self._sess = tf.Session()
        self._states = tf.placeholder(tf.float32, (None, num_input), name="states")

        self._corrupt_prob = tf.placeholder(tf.float32, [1])
        self._current_input = self._corrupt(self._states) * self._corrupt_prob + self._states * (1 - self._corrupt_prob)


        self._init_learning_rate = init_learning_rate
        self._min_learning_rate = min_learning_rate
        self._learning_rate_N_max = learning_rate_N_max
        self._learning_rate = tf.placeholder(tf.float32, shape=[])

        # policy parameters
        with tf.variable_scope("policy_params"):
            self._mu_theta = tf.get_variable("mu_theta", [32, 1],
                                             initializer=tf.zeros_initializer())
            self._sigma_theta = tf.get_variable("sigma_theta", [32],
                                                initializer=tf.zeros_initializer())

        # neural featurizer encoder
        with tf.variable_scope("autoencoder_params"):
            self._W1 = tf.get_variable("W1", [num_input, 32],
                                       initializer=tf.random_normal_initializer())
            self._b1 = tf.get_variable("b1", [32],
                                       initializer=tf.constant_initializer(0))
            self._phi = tf.matmul(self._current_input, self._W1) + self._b1

            # neural featurizer decoder
            self._W2 = tf.get_variable("W2", [32, 2],
                                       initializer=tf.random_normal_initializer(stddev=0.1))
            self._b2 = tf.get_variable("b2", [2],
                                       initializer=tf.constant_initializer(0))
            self._reconstruction = tf.nn.tanh(tf.matmul(self._phi, self._W2) + self._b2)

        self._mu = tf.matmul(self._phi, self._mu_theta)
        self._sigma = tf.reduce_sum(self._sigma_theta)
        self._sigma = tf.exp(self._sigma)

        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)

        self._discounted_rewards = tf.placeholder(tf.float32, (None, 1), name="discounted_rewards")
        self._taken_actions = tf.placeholder(tf.float32, (None, 1), name="taken_actions")

        # we'll get the policy gradient by using -log(pdf), where pdf is the PDF of the Normal distribution
        self._loss = -tf.log(tf.sqrt(1/(2 * np.pi * self._sigma**2)) * tf.exp(-(self._taken_actions - self._mu)**2/(2 * self._sigma**2))) * self._discounted_rewards

        policy_param_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy_params")
        self._train_op = self._optimizer.minimize(self._loss, var_list=policy_param_vars)

        # self._recon_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self._states, predictions=self._reconstruction))
        self._recon_loss = tf.sqrt(tf.reduce_mean(tf.square(self._states - self._reconstruction)))

        autoencoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "autoencoder_params")
        self._recon_train_op = self._optimizer.minimize(self._recon_loss, var_list=autoencoder_vars)

        self._sess.run(tf.global_variables_initializer())

        self._num_input = num_input
        self._shuffle = shuffle
        self._batch_size = batch_size
        # rollout buffer
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        # record reward history for normalization
        self._all_rewards = []
        self._max_reward_length = 1000000
        self._discount_factor = 0.99

        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self._scaler = sklearn.preprocessing.StandardScaler()
        self._scaler.fit(observation_examples)

    def sample_action(self, system_state):
        system_state = self._scaler.transform(system_state.reshape(1, -1))
        mu, sigma = self._sess.run([self._mu, self._sigma], feed_dict={
            self._states: np.reshape(system_state, (1, self._num_input)),
            self._corrupt_prob: [0]
        })
        action = np.random.normal(mu, sigma)
        action = np.clip(action, self._env.action_space.low[0], self._env.action_space.high[0])
        return action, sigma

    def store_rollout(self, state, action, reward):
        self._action_buffer.append(action)
        self._reward_buffer.append(reward)
        self._state_buffer.append(state)

    def update_model(self, iteration):
        N = len(self._reward_buffer)
        r = 0 # use discounted reward to approximate Q value

        # compute discounted future rewards
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            # future discounted reward from now on
            r = self._reward_buffer[t] + self._discount_factor * r
            discounted_rewards[t] = r

        # reduce gradient variance by normalization
        self._all_rewards += discounted_rewards.tolist()
        self._all_rewards = self._all_rewards[:self._max_reward_length]
        discounted_rewards -= np.mean(self._all_rewards)
        discounted_rewards /= np.std(self._all_rewards)

        learning_rate = self._gen_learning_rate(iteration, l_max=self._init_learning_rate,
                                                l_min=self._min_learning_rate, N_max=self._learning_rate_N_max)

        all_samples = []
        for t in range(N-1):
            state  = np.reshape(np.array(self._state_buffer[t]), self._num_input)
            action = self._action_buffer[t][0]
            reward = [discounted_rewards[t]]
            sample = [state, action, reward]
            all_samples.append(sample)
        if self._shuffle:
            np.random.shuffle(all_samples)

        batches = list(self._minibatches(all_samples, batch_size=self._batch_size))

        recon_losses = []
        for b in range(len(batches)):
            batch = batches[b]
            states = [row[0] for row in batch]
            _, recon = self._sess.run([self._recon_train_op, self._recon_loss], {
                self._states: states,
                self._learning_rate: learning_rate,
                self._corrupt_prob: [1.0]
            })
            recon_losses.append(recon)
        # print("mean recon loss: %s" % np.mean(recon_losses))

        for b in range(len(batches)):
            batch = batches[b]
            states = [row[0] for row in batch]
            actions = [row[1] for row in batch]
            rewards = [row[2] for row in batch]

            self._sess.run([self._train_op], {
                self._states:             states,
                self._taken_actions:      actions,
                self._discounted_rewards: rewards,
                self._learning_rate:      learning_rate,
                self._corrupt_prob: [0]
            })

        self._clean_up()

    def _minibatches(self, samples, batch_size):
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def _gen_learning_rate(self, iteration, l_max, l_min, N_max):
        if iteration > N_max:
            return l_min
        alpha = 2 * l_max
        beta = np.log((alpha / l_min - 1)) / N_max
        return alpha / (1 + np.exp(beta * iteration))

    def _clean_up(self):
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []

    def _corrupt(self, x):
        return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                                        minval=0,
                                                        maxval=2,
                                                        dtype=tf.int32), tf.float32))
