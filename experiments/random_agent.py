import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
from gym_insurance.envs.insurenv import InsurEnv

VALUE_COLUMN = "valor_indenização"
BUDGET = 99999999


train_data = pd.read_csv("../../data/processed/psr_train_set.csv")
train_data = train_data.query("valor_indenização!=1147131.5")

test_data = pd.read_csv("../../data/processed/psr_test_set.csv")
test_data = test_data.query("valor_indenização!=1147131.5")

state_columns = train_data.columns[5:-1]
env = InsurEnv(train_data, VALUE_COLUMN, state_columns, BUDGET)


class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.env.reset()

    def act(self, state):
        return self.env.action_space.sample()


agent = RandomAgent(env)
app = dash.Dash()

app.layout = html.Div(
    children=[
        html.H1(children="Rewards over time"),
        dcc.Graph(
            id="rewards-over-time",
            figure={
                "data": [go.Scatter(x=[], y=[], mode="lines+markers")],
                "layout": go.Layout(
                    xaxis={"title": "Time step"}, yaxis={"title": "Reward"}
                ),
            },
        ),
    ]
)

# Define the callback function to update the rewards plot
@app.callback(
    Output("rewards-over-time", "figure"), [Input("rewards-over-time", "figure")]
)
def update_graph(figure):
    rewards = env.rewards
    x = list(range(len(rewards)))
    y = rewards

    figure["data"][0]["x"] = x
    figure["data"][0]["y"] = y

    return figure


# In the loop for each episode
state = env.reset()
done = False
while not env.done:
    action = agent.act(state)
    state, reward, done, info = env.step(action)

    # Update the rewards plot
    app.callback(
        Output("rewards-over-time", "figure"), [Input("rewards-over-time", "figure")]
    )()

if __name__ == "__main__":
    app.run_server(debug=True)
