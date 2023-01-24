import random
import time
from collections import deque
from tensorflow.python.framework.errors_impl import InvalidArgumentError
import numpy as np
import tqdm
from gym_insurance.envs.utils import ModifiedTensorBoard

MODEL_NAME = "br_crop_insurance"


class DQNAgent:
    def __init__(self, env, model):
        self.env = env
        self.env.reset()
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.target_update_counter = 0
        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time()))
        )
        self.model = model
        self.tensorboard.set_model(model)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # Start training only if certain number of samples is already saved
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(
                state,
                target_f,
                batch_size=batch_size,
                verbose=1,
                #        callbacks=[self.tensorboard],
            )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict(self, new_data):
        """
        predict the decision (approve or reject) for new claims data
        :param new_data: new claims data as a pandas dataframe
        :return: a list of decisions (1 for approve, 0 for reject)
        """
        new_data = new_data[self.env.state_columns].values
        decisions = np.argmax(self.model.predict(new_data), axis=1)
        return decisions

    def predict_proba(self, new_data):
        """
        predict the probability of approving a claim for new claims data
        :param new_data: new claims data as a pandas dataframe
        :return: a list of probabilities of approving a claim
        """
        new_data = new_data[self.env.state_columns].values
        proba = self.model.predict(new_data)
        return proba[:, 1]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    @property
    def _shape_state(self):
        return np.reshape(self.env.state, [1, self.env.observation_space.shape[0]])

    @property
    def _get_sa(self):
        self.env.reset()
        action = self.act(self._shape_state)
        return self._shape_state, action

    def fit(self, episodes, min_replay_memory_size, min_reward, batch_size):
        assert min_replay_memory_size > 2
        for i in tqdm.tqdm(range(1, episodes + 1), ascii=True, unit="episodes"):
            self.env.reset()
            self.min_replay_memory_size = min_replay_memory_size
            state, action = self._get_sa
            while not self.env.done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self._shape_state
                self.remember(state, action, reward, next_state, done)
                self.replay(batch_size)

            # Update target network counter every episode
            if done:
                self.target_update_counter += 1
                try:
                    self.tensorboard.update_stats(
                        step=i,
                        reward=sum(self.env.rewards),
                        min_reward=min(self.env.rewards),
                        approved_pct=self.env.approved_pct,
                        budget_pct=self.env.budget.pct_budget,
                        loss=self.model.history.history["loss"],
                    )
                except:
                    pass

    def transform(self, data):
        self.env.reset()
        self.test_data = data
        for i in data.iterrows():
            values = list(i) + [self.env.pct_budget, self.env.approved / self.env.steps]

        return np.reshape(values,  [1, data.shape[1]])
        # assert data.shape[1] == self.state_size
        #return self.predict(new_data=data)