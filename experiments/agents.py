import random
import time
from collections import deque

import numpy as np
import pandas as pd
from gym_insurance.envs.insurenv import InsurEnv
from gym_insurance.envs.utils import ModifiedTensorBoard
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

MODEL_NAME = "br_crop_insurance"

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time()))
        )
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        self.tensorboard.set_model(model)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(
                state,
                target_f,
                epochs=1,
                verbose=0,
                shuffle=False,
                callbacks=[self.tensorboard] if self.env.done else None,
            )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
