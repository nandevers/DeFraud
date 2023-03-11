import plotly.graph_objs as go

# Sample data for the number of steps and total reward
steps = [0, 1, 2, 3, 4, 5]
total_reward = [0, 2, 4, 6, 8, 10]


def rewards_line_plot(steps, total_reward):
    # Create a trace for the line chart
    trace = go.Scatter(x=steps, y=total_reward, mode="lines", name="Reward")

    # Set the layout for the chart
    layout = go.Layout(
        title="Total Reward vs. Number of Steps",
        xaxis=dict(title="Steps"),
        yaxis=dict(title="Total Reward"),
    )

    # Create the figure and plot the chart
    return go.Figure(data=[trace], layout=layout)
