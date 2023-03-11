import streamlit as st
import pandas as pd
from chart_utils import rewards_line_plot
import plotly.express as px


st.title("Experiments")


def read_results(strategy = 'random'):
    return pd.read_parquet(f"data/model_results/{strategy}/_parquet/")


random_tab, dqn_tab = st.tabs(["Random", "DQN Agent"])

with random_tab:
    st.write("A random strategy applied to the environment ")
    results = read_results().loc[:, ["steps", "rewards", "budget", 'initial_budget']].sort_values("steps")

    st.plotly_chart(
        rewards_line_plot(steps=results["steps"], total_reward=results["rewards"])
    )
    st.plotly_chart(
        rewards_line_plot(steps=results["steps"], total_reward=results["budget"])
    )
    st.plotly_chart(
        px.line(
        results.loc[:,['steps', 'rewards', 'initial_budget']].melt(id_vars=['initial_budget', 'steps']),
                       x="steps", 
                       y="value", 
                       color='initial_budget')
    )

    st.dataframe(
        read_results().pivot_table(
            index="decision",
            columns="initial_budget",
            values="nr_documento_segurado",
            aggfunc="count",
        )
    )
    st.dataframe(
        read_results().pivot_table(
            index="decision", columns="initial_budget", values="steps", aggfunc="max"
        )
    )
    st.dataframe(
        read_results().pivot_table(
            index="decision", 
            columns="initial_budget", 
            values="valor_indenizacao", 
            aggfunc=["min", "max", "mean", "std"],

        )
    )

with dqn_tab:
    st.write("A random strategy applied to the environment ")
    model_strategy = 'baseline'
    results = read_results(model_strategy).loc[:, ["steps", "rewards", "budget"]].sort_values("steps")

    st.plotly_chart(
        rewards_line_plot(steps=results["steps"], total_reward=results["rewards"])
    )
    st.plotly_chart(
        rewards_line_plot(steps=results["steps"], total_reward=results["budget"])
    )
    st.plotly_chart(
        px.line(
        read_results(model_strategy).loc[:,['steps', 'rewards', 'initial_budget']].melt(id_vars=['initial_budget', 'steps']),
                       x="steps", 
                       y="value", 
                       color='initial_budget')
    )
    st.dataframe(
        read_results(model_strategy).pivot_table(
            index="decision",
            columns="initial_budget",
            values="nr_documento_segurado",
            aggfunc="count",
        )
    )
    st.dataframe(
        read_results(model_strategy).pivot_table(
            index="decision", columns="initial_budget", values="valor_indenizacao", aggfunc=["min", "max", "mean", "std"]
        )
    )