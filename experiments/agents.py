import random

import tensorflow as tf
from collections import deque
from tensorflow.python.framework.errors_impl import InvalidArgumentError
import numpy as np
import tqdm

MODEL_NAME = "br_crop_insurance"
UPDATE_TARGET_EVERY = 5
MIN_MEMORY = 500
MEMORY_SIZE = 4000

np.random.seed(1)
random.seed(1)


class DQNAgent:
    def __init__(
        self,
        env,
        model,
        memory_size=MEMORY_SIZE,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.001,
        epsilon_decay=0.995,
    ):
        self.env = env
        self.env.reset()
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_counter = 0
        self.model = model

        # Target network
        self.target_model = model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_qs(self, state):
        if state.sum() == 0 and state.shape == (self.state_size,):
            state = state.reshape(1, self.state_size)
        return self.model.predict(state)[0]

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.get_qs(state)
        return np.argmax(act_values)

    # FIXME: shouldnt this be train instead?
    def replay(self, batch_size, terminal_state, step):
        assert batch_size < MIN_MEMORY
        if len(self.memory) < MIN_MEMORY:
            return

        minibatch = random.sample(self.memory, batch_size)
        current_states = np.array(
            [
                transition[0].reshape(self.state_size, 1).transpose()
                for transition in minibatch
            ]
        )
        current_states = np.concatenate(current_states, axis=0)
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array(
            [
                transition[3].reshape(self.state_size, 1).transpose()
                for transition in minibatch
            ]
        )
        new_current_states = np.concatenate(new_current_states, axis=0)
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []
        # Now we need to enumerate our batches
        for index, (
            current_state,
            action,
            reward,
            new_current_state,
            done,
        ) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if reward is None:
                reward = 0

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state.reshape(self.state_size, 1).transpose())
            X_ = np.concatenate(X, axis=0)
            y.append(current_qs.reshape(self.action_size, 1).transpose())
            y_ = np.concatenate(y, axis=0)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            X_,
            y_,
            #            np.array(X),
            #            np.array(y),
            batch_size=batch_size,
            verbose=1,
            shuffle=False,
        )

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def predict(self, new_data):
        """
        predict the decision (approve or reject) for new claims data
        :param new_data: new claims data as a pandas dataframe
        :return: a list of decisions (1 for approve, 0 for reject)
        """
        new_data = new_data[self.env.state_columns].values
        decisions = np.argmax(self.model.predict(new_data), axis=1)
        return decisions

    def predict_proba(self, new_data):
        """
        predict the probability of approving a claim for new claims data
        :param new_data: new claims data as a pandas dataframe
        :return: a list of probabilities of approving a claim
        """
        new_data = new_data[self.env.state_columns].values
        proba = self.model.predict(new_data)
        return proba[:, 1]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    @property
    def _shape_state(self):
        return np.reshape(self.env.state, [1, self.env.observation_space.shape[0]])

    @property
    def _get_sa(self):
        self.env.reset()
        action = self.act(self._shape_state)
        return self._shape_state, action


class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.env.reset()

    def act(self):
        return self.env.action_space.sample()
