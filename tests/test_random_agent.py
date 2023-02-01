import numpy as np
import pandas as pd
from gym_insurance.envs.insurenv import InsurEnv

VALUE_COLUMN = "valor_indenização"
BUDGET = 99999999


train_data = pd.read_csv("../../data/processed/psr_train_set.csv")
train_data = train_data.query("valor_indenização!=1147131.5")

state_columns = train_data.columns[5:-1]
env = InsurEnv(train_data, VALUE_COLUMN, state_columns, BUDGET)
env.reset()
while not env.done:
    action = np.random.randint(0, 2)
    observation, reward, done, info = env.step(action)
    print(reward, action)
