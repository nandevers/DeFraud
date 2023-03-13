import numpy as np
import pandas as pd

from gym_insurance.envs.insurenv import InsurEnv
from experiments.agents import DQNAgent
from experiments.architectures import build_baseline_model

MODEL_NAME = "br_crop_insurance"
UPDATE_TARGET_EVERY = 5
MIN_MEMORY = 1000
MEMORY_SIZE = 4000
batch_size = 300
EPISODES = 100
LEARNING_RATE = 0.01
UPDATE_TARGET_EVERY = 10

data = pd.read_csv("data/processed/psr_train_set.csv").query(
    "valor_indenizacao==valor_indenizacao"
)

value_column = "valor_indenizacao"
state_columns = data.columns.to_list()[5:-1]
env = InsurEnv(
    data=data,
    index_column="id_proposta",
    value_column=value_column,
    state_columns=state_columns,
    budget=999999*5,
)

model = build_baseline_model(
    state_size=len(state_columns) + 2,
    action_size=env.action_space.n,
    learning_rate=LEARNING_RATE,
)
agent = DQNAgent(
    env,
    model,
    memory_size=MEMORY_SIZE,
    min_memory_size=MIN_MEMORY,
    update_target_every=UPDATE_TARGET_EVERY,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.001,
    epsilon_decay=0.995,
)

_model_results = []
for e in range(1, EPISODES + 1):
    done = False
    state = env.reset()
    while done is False:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        # reward = reward if not done else -10
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
        f"data/model_results/baseline/_csv/{model_results_file_name}_{batch_size}_results.csv",
        index=False,
    )
