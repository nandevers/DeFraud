import numpy as np
import pandas as pd

from experiments.agents import RandomAgent
from gym_insurance.envs.insurenv import InsurEnv

VALUE_COLUMN = "valor_indenizacao"
BUDGET_LIST = [99999, 999999, 9999999]


train_data = pd.read_csv("data/processed/psr_train_set.csv")
train_data = train_data.query("valor_indenizacao!=1147131.5")
train_data = train_data.query("valor_indenizacao==valor_indenizacao")

state_columns = train_data.columns[5:]

for b in BUDGET_LIST:
    env = InsurEnv(
        data=train_data,
        index_column="id_proposta",
        value_column=VALUE_COLUMN,
        state_columns=state_columns,
        budget=b,
    )
    train_data["id_proposta"]
    state = env.reset()
    done = False
    agent = RandomAgent(env)
    while not env.done:
        action = agent.act()
        state, reward, done, info = env.step(action)
    env.results.columns.to_list()
    env.results.to_parquet(f"data/model_results/_parquet/random_{b}_results.parquet")
    env.results.to_csv(f"data/model_results/_csv/random_{b}_results.csv", index=False)
