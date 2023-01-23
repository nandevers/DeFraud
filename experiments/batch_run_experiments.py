import pandas as pd
from gym_insurance.envs.insurenv import InsurEnvo

data = pd.read_csv("data/processed/psr_train_set.csv")
value_column = "valor_indenização"
state_columns = data.columns[5:-1]
budget = 10000
env = InsurEnv(data, value_column, state_columns, budget)
agent = DQNAgent(env)


# agent.load("./save/cartpole-dqn.h5")
done = False
batch_size = 32

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    for time in range(500):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(
                "episode: {}/{}, score: {}, e: {:.2}".format(
                    e, EPISODES, time, agent.epsilon
                )
            )
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
