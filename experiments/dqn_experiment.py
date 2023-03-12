import numpy as np
import pandas as pd

from gym_insurance.envs.insurenv import InsurEnv
from experiments.agents import DQNAgent
from experiments.architectures import build_baseline_model

LEARNING_RATE = 0.001
data = pd.read_csv("data/processed/psr_train_set.csv").query(
    "valor_indenizacao==valor_indenizacao"
)
#data["first_digit"] = data["valor_indenizacao"].astype(str).str[0]
#data["first_digit"] = data["first_digit"].astype(int)
#data["second_digit"] = (
#    data["valor_indenizacao"].astype(str).replace("\.", "", regex=True).str[1]
#)
#data["second_digit"] = data["second_digit"].astype(int)

value_column = "valor_indenizacao"
state_columns = data.columns.to_list()[5:-1] #+ ["first_digit", "second_digit"]
env = InsurEnv(
    data=data,
    index_column="id_proposta",
    value_column=value_column,
    state_columns=state_columns,
    budget=999999*5,
)
batch_size = 100  # Pick from 1 to 1000
EPISODES = 500

model = build_baseline_model(
    state_size=len(state_columns) + 2,
    action_size=env.action_space.n,
    learning_rate=LEARNING_RATE,
)
agent = DQNAgent(env, model)

model.summary()

_model_results = []
for e in range(1, EPISODES + 1):
    done = False
    state = env.reset()

    while done is False:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        # FIX ME: What is this about? Correct?
        if len(agent.memory) > batch_size:
            agent.replay(batch_size, done, env.steps)

        _results = env.results.copy()
        _results["episodes"] = e
        _model_results.append(_results)

    model_results = pd.concat(_model_results, axis=0)
    model_results_file_name = f"dqnagent_base_line_{LEARNING_RATE}"

    model_results.to_parquet(
        f"data/model_results/baseline/_parquet/{model_results_file_name}_{batch_size}_results.parquet"
    )
    model_results.to_csv(
        f"data/model_results/baseline/_csv/{model_results_file_name}__{batch_size}_results.csv",
        index=False,
    )
