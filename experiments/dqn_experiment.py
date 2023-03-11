import numpy as np
import pandas as pd
from gym_insurance.envs.insurenv import InsurEnv
from experiments.agents import DQNAgent
from experiments.architectures import build_baseline_model

LEARNING_RATE = 0.01
data = pd.read_csv("data/processed/psr_train_set.csv").query(
    "valor_indenizacao==valor_indenizacao"
)

value_column = "valor_indenizacao"
state_columns = data.columns[5:-1]
budget = 999999
env = InsurEnv(
    data=data,
    index_column="id_proposta",
    value_column=value_column,
    state_columns=state_columns,
    budget=budget,
)
batch_size = 300
EPISODES = 10
model = build_baseline_model(
    state_size=len(state_columns) + 2,
    action_size=env.action_space.n,
    learning_rate=0.001,
)
agent = DQNAgent(env, model)

model.summary()


for e in range(EPISODES):
    done = False
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    while done is False:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        print("episode: {}/{},  e: {:.2}".format(e, EPISODES, agent.epsilon))
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

model_results_file_name = f"dqnagent_base_line{LEARNING_RATE}"
env.results.to_parquet(
    f"data/model_results/baseline/_parquet/{model_results_file_name}_results.parquet"
)
env.results.to_csv(
    f"data/model_results/baseline/_csv/{model_results_file_name}_results.csv",
    index=False,
)
