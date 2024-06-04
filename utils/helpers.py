"""Helpers for the Example II."""

import matplotlib.pyplot as plt
import network_diffusion as nd
import numpy as np
import pandas as pd


def get_mean_log(logs: list[pd.DataFrame]):
    """Return df with averaged statistics from the bunch of simulations."""
    base_df = logs[0].copy()
    for _df in logs[1:]:
        base_df = base_df.add(_df, fill_value=0)
    return base_df / len(logs)


def get_std_log(logs: list[pd.DataFrame]):
    """Return df with std. dev. of statistics from the bunch of simulations."""
    mean_df = get_mean_log(logs)
    base_df = logs[0].copy() * 0
    for _df in logs:
        base_df += np.power((_df - mean_df), 2)
    return np.sqrt(base_df / len(logs))


def aggregate_results(all_logs: list[nd.Logger]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return df with averaged statistics from the bunch of simulations."""
    all_logs_contagion = [log._global_stats_converted["contagion"] for log in all_logs]
    all_logs_awareness = [log._global_stats_converted["awareness"] for log in all_logs]
    mean_contagion = get_mean_log(all_logs_contagion)
    mean_awareness = get_mean_log(all_logs_awareness)
    mean_core = pd.DataFrame([mean_contagion["I"], mean_awareness["A"]]).T
    std_contagion = get_std_log(all_logs_contagion)
    std_awareness = get_std_log(all_logs_awareness)
    std_core = pd.DataFrame([std_contagion["I"], std_awareness["A"]]).T
    return mean_core, std_core


def visualise_spread(mean_result: pd.DataFrame, std_results: pd.DataFrame) -> None:
    """Visualise the Experiment II."""
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10,5)
    colour = "red"
    # extract stats to np arrays
    _i = mean_result["I"].to_numpy()
    _i_std = std_results["I"].to_numpy()
    _a = mean_result["A"].to_numpy()
    _a_std = std_results["A"].to_numpy()
    _x = np.arange(0, len(_i))
    # draw curves
    ax[0].plot(_x, _i, label=rf"Infected", color=colour)
    ax[0].fill_between(_x, _i-_i_std, _i+_i_std, alpha=0.1, color=colour)
    ax[1].plot(_x, _a, label=rf"Aware", color=colour)
    ax[1].fill_between(_x, _a-_a_std, _a+_a_std, alpha=0.1, color=colour)
    # adjust legend and other markings
    for a in ax:
        a.grid(True)
        a.set_xlabel("Epoch")
        a.set_xticks(np.arange(0, _x.max()+1, 5))
        a.set_ylabel("nb of Agents")
        ax[0].legend(loc="upper right")
        ax[1].legend(loc="lower right")
    # plot the result
    fig.tight_layout()
