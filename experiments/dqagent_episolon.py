import time
import pandas as pd
import numpy as np
from gym_insurance.envs.insurenv import InsurEnv
from agents import DQNAgent

#from experiments.agents import DQNAgent
from gym_insurance.envs.utils import ModifiedTensorBoard


AGGREGATE_STATS_EVERY = 5
MODEL_NAME = "br_crop_insurance"

tensorboard = ModifiedTensorBoard(
    log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time()))
)


data = pd.read_csv("../../data/processed/psr_train_set.csv")
data = data.query("valor_indenização!=1147131.5")
value_column = "valor_indenização"
state_columns = data.columns[5:-1]
budget = 1000000
env = InsurEnv(data, value_column, state_columns, budget)
env.reset()



dir(env)
agent = DQNAgent(env)
done = False
batch_size = 32
EPISODES = 2
MIN_REWARD = -200

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    for t in range(500):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        print(env.results)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            average_reward = sum(env.rewards[-AGGREGATE_STATS_EVERY:]) / len(
                env.rewards[-AGGREGATE_STATS_EVERY:]
            )
            min_reward = min(env.rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(env.rewards[-AGGREGATE_STATS_EVERY:])
            tensorboard.set_model(agent.model)

            agent.tensorboard.update_stats(
                step=t,
                reward_avg=average_reward,
                reward_min=min_reward,
                reward_max=max_reward,
                approved_pct=env.approved_pct,
                pct_budget=env.budget.pct_budget,
                epsilon=agent.epsilon,
            )

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:

                agent.model.save(
                    f"logs/models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model"
                )
                print(
                    "episode: {}/{}, score: {}, e: {:.2}".format(
                        e, EPISODES, t, agent.epsilon
                    )
                )
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)


from seaborn import histplot
from matplotlib import pyplot as plt
import tensorflow as tf

with env.tensorboard.writer.as_default():
    histplot(env.results, x="valor_indenização", hue="decision")
    plt.show()
    x = env.results.query("decision==1")["valor_indenização"].values
    tf.summary.histogram("Approved", x, step=env.episodes, description="Appoved Values")

    x = env.results.query("decision==0")["valor_indenização"].values
    tf.summary.histogram(
        "Rejected", x, step=env.episodes, description="Rejected Values"
    )
