# from agents import DQNAgent
import time

import numpy as np
import pandas as pd

from experiments.agents import DQNAgent
from experiments.architectures import build_model
from agents import DQNAgent
from architectures import build_model
from gym_insurance.envs.insurenv import InsurEnv

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
    env.observation_space.shape[0], env.action_space.n, learning_rate=0.1
)
model.summary()

dqagent = DQNAgent(env, model)
dqagent.fit(episodes=10, min_replay_memory_size=3, min_reward=3, batch_size=5)

dqagent.env.results.to_csv(dqagent.tensorboard.log_dir + '/train_results.csv' , index = False)


results = []
for i, j in enumerate(test_data[dqagent.env.state_columns].iterrows()):
    values = list(j[1].values)
    if i ==0:
        values = values + [0.0 , 0.0]
        values = np.reshape(values, [1, 77])
    else:
        values = values + [dqagent.env.budget.pct_budget, dqagent.env.approved/dqagent.env.steps]
        values = np.reshape(values, [1, 77])
    action = dqagent.act(values)
    test_data.loc[j[0], 'decisions'] = action
    results.append(action)
    dqagent.env.step(action)
    if dqagent.env.done:
        break
dqagent.env.results
dqagent.env.steps


dqagent.model.predict(test_data[dqagent.env.state_columns].iloc[0].values)
dqagent.transform(test_data)
