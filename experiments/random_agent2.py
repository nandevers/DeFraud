import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
from gym_insurance.envs.insurenv import InsurEnv

VALUE_COLUMN = "valor_indenização"
BUDGET = 99999999


class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.env.reset()

    def act(self):
        return np.random.randint(0, 2)


if __name__ == "__main__":
    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Graph(id="histogram", style={"display": "inline-block"}),
            dcc.Graph(id="live-update-graph", style={"display": "inline-block"}),
            dcc.Interval(
                id="interval-component",
                interval=1 * 1000,  # in milliseconds
                n_intervals=0,
            ),
        ]
    )

    @app.callback(
        [
            Output("histogram", "figure"),
            Output("live-update-graph", "figure"),
            Input("interval-component", "n_intervals"),
        ],
        prevent_initial_call=True,
    )
    def update_graph(steps):
        traces = [go.Scatter(x=list(range(steps)), y=env.rewards, mode="lines+markers")]
        fig = {
            "data": traces,
            "layout": go.Layout(xaxis={"title": "Step"}, yaxis={"title": "Reward"}),
        }
        hist_fig = go.Figure(
            data=[
                go.Histogram(
                    x=list(env.budget.decisions.keys()),
                    marker=dict(color=list(env.budget.decisions.values())),
                    nbins=50,
                )
            ]
        )
        return hist_fig, fig

    train_data = pd.read_csv("../../data/processed/psr_train_set.csv").query(
        "valor_indenização!=1147131.5"
    )

    state_columns = train_data.columns[5:-1]
    env = InsurEnv(train_data, VALUE_COLUMN, state_columns, BUDGET)
    env.reset()
    agent = RandomAgent(env)
    while not env.done:
        action = agent.act()
        env.step(action)
        update_graph(env.steps)

    app.run(port=8050, debug=True)
