import pandas as pd
import plotly.express as px
import streamlit as st
from benfordslaw import benfordslaw as bf

from gym_insurance.envs.benford import Benford
from chart_utils import rewards_line_plot
import numpy as np

st.title("Experiments")


@st.cache_resource
def read_results(strategy="random"):
    return (
        pd.read_parquet(f"data/model_results/{strategy}/_parquet/")
        .query("steps==steps")
        .drop_duplicates()
    )


def read_cols(strategy="random"):
    return read_results(strategy=strategy).columns.to_list()


random_tab, dqn_tab = st.tabs(["Random", "Baseline"])

with random_tab:
    st.write("A random strategy applied to the environment ")
    results = (
        read_results()
        .loc[:, ["steps", "rewards", "budget", "initial_budget"]]
        .sort_values("steps")
    )

    st.plotly_chart(
        px.line(
            results.loc[:, ["steps", "rewards", "initial_budget"]].melt(
                id_vars=["initial_budget", "steps"]
            ),
            x="steps",
            y="value",
            color="initial_budget",
        )
    )
    st.plotly_chart(
        px.histogram(
            results, x="rewards", color="initial_budget", facet_col="initial_budget"
        ).update_xaxes(matches=None)
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


## FOR DQN TAB


def results_by_decision(model_strategy, decision=None):
    if decision is None:
        return read_results(model_strategy).loc[:, ["valor_indenizacao"]]
    else:
        return (
            read_results(model_strategy)
            .query(f"decision=={decision} and episodes=={st.session_state.episodes}")
            .loc[:, ["valor_indenizacao"]]
        )


def bf_analysis(d=1):
    overall =  bf(d)
    approved = bf(d)
    rejected = bf(d)

    overall.fit(results_by_decision(model_strategy, None))
    approved.fit(results_by_decision(model_strategy, 1))
    rejected.fit(results_by_decision(model_strategy, 0))
    return overall, approved, rejected


def plot_benford(d=1):
    overall, approved, rejected = bf_analysis()
    overall_fig, _ = overall.plot()
    approved_fig, _ = approved.plot()
    rejected_fig, _ = rejected.plot()
    return overall_fig, approved_fig, rejected_fig


def melt_benford(d=1):
    col = "percentage_emp"
    bf0, bf1, bf2 = bf_analysis()
    df = [pd.DataFrame(b.results[col]).iloc[:, 1] for b in bf_analysis(d)]
    df = pd.concat(df, axis=1)
    df.columns = ["overall", "approved", "rejected"]
    return df


with dqn_tab:
    model_strategy = "baseline"
    results = (
        read_results(model_strategy)
        .loc[:, ["episodes", "steps", "rewards", "budget"]]
        .sort_values(["episodes", "steps"])
    )

    with st.sidebar:
        st.session_state.episodes = st.selectbox(
            f"Select sample by episode",
            options=results.episodes.unique(),
        )

    if st.button("Clear All"):
        # Clears all st.cache_resource caches:
        st.cache_resource.clear()


    st.plotly_chart(
        px.line(results.query("episodes>0"), x="steps", y="rewards", color="episodes")
    )

    st.markdown("Approved /  Rejected by Esisode: Total and Monetary Values")
    rewards_summary = (
        read_results(model_strategy)
        .groupby("episodes")["rewards"]
        .agg(["min", "mean", "max", "sum", "std", "count"])
        .reset_index()
    )
    rewards_summary["cum_sum"] = rewards_summary["sum"].cumsum()
    st.dataframe(rewards_summary.sort_values('sum', ascending=False))
    st.plotly_chart(px.line(data_frame=rewards_summary, x="episodes", y="cum_sum"))

    st.dataframe(
        read_results(model_strategy).pivot_table(
            index="episodes",
            values="decision",
            aggfunc=["min", "mean", "max", "std", "count"],
        ).reset_index()
    )

    max_reward = read_results(model_strategy).rewards.max()
    st.metric(label="Max Reward", value=np.round(max_reward))

    with st.expander("Open to see results"):
        st.dataframe(
            # read_results(model_strategy).groupby(["episodes", "decision"])['valor_indenizacao'].apply(lambda x: list(Benford(3).chi_test(x)))
            read_results(model_strategy).pivot_table(
                index="episodes",
                columns="decision",
                values="valor_indenizacao",
                aggfunc=lambda x: any(Benford(1).chi_test(x)),
            )
        )

    st.markdown("### Analysis of Digits with Benford's Distribution")
    bf_dist = melt_benford().melt().reset_index()
    st.plotly_chart(
        px.bar(bf_dist, x=bf_dist["index"], y="value", color="variable"),
        use_container_width=True,
    )

    overall, approved, rejected = plot_benford()
    col1, col2, col3 = st.columns(3)
    


    o, a, r = bf_analysis()
    with col1:
        decision = None
        st.header("Overall")
        st.write(o.results["P_significant"])
        p1, p2, p3 = Benford(3).plot(results_by_decision(model_strategy, decision=decision))
        st.pyplot(p1)
        st.pyplot(p2)
        st.pyplot(p3)

    with col2:
        decision = 1
        st.header("Approved")
        st.write(a.results["P_significant"])
        p1, p2, p3 = Benford(3).plot(results_by_decision(model_strategy, decision=decision))
        st.pyplot(p1)
        st.pyplot(p2)
        st.pyplot(p3)
 

    with col3:
        decision = 0
        st.header("Rejected")
        st.write(r.results["P_significant"])
        p1, p2, p3 = Benford(3).plot(results_by_decision(model_strategy, decision=decision))
        st.pyplot(p1)
        st.pyplot(p2)
        st.pyplot(p3)


