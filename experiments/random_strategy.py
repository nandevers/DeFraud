class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.env.reset()

    def act(self):
        return self.env.action_space.sample()
