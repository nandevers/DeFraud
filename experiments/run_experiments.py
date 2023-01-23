# from agents import DQNAgent
import time

import numpy as np
import pandas as pd
from experiments.agents import DQNAgent
from gym_insurance.envs.insurenv import InsurEnv
from experiments.architectures import build_model

VALUE_COLUMN = "valor_indenização"
BUDGET = 10000000


train_data = pd.read_csv("../../data/processed/psr_train_set.csv")
train_data = train_data.query("valor_indenização!=1147131.5")

test_data = pd.read_csv("../../data/processed/psr_test_set.csv")
test_data = test_data.query("valor_indenização!=1147131.5")


state_columns = train_data.columns[5:-1]
env = InsurEnv(train_data, VALUE_COLUMN, state_columns, BUDGET)
env.reset()


model = build_model(
    "experiments/architectures/architecture_1.yaml",
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
)
