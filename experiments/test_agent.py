import time
import pandas as pd
import numpy as np
from gym_insurance.envs.insurenv import InsurEnv
from gym_insurance.envs.utils import ModifiedTensorBoard
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


#from experiments.agents import DQNAgent
from agents import DQNAgent


def build_model(state_size, action_size, learning_rate):
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(action_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))
    return model


data = pd.read_csv("../../data/processed/psr_train_set.csv")
data = data.query("valor_indenização!=1147131.5")
value_column = "valor_indenização"
state_columns = data.columns[5:-1]
budget = 10000000
env = InsurEnv(data, value_column, state_columns, budget)
env.reset()

model = build_model(env.observation_space.shape[0], env.action_space.n, learning_rate=.1)
model.summary()

dqagent = DQNAgent(env, model)
dqagent.fit(episodes=10, min_replay_memory_size=3, min_reward=3, batch_size=5)
print(dqagent.env.episodes)