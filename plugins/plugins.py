import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import warnings
import time

from my_fun.my_fun import *  # Or from daily_report.daily_report import daily_report
from parse.parse import *  # Or from parse.parse import parse

warnings.filterwarnings('ignore')


def plot(df_session):
    ax1.clear()

    ####################################################################################################################

    # Plotting style
    ms = 3  # (mpl default for plt.plot=6 and for plt.scatter=6**2)
    lw = 1.5  # mpl default

    ####################################################################################################################

    # SUMMARY VARIABLES
    # Trials
    trials = len(df_session)
    trials_left = df_session.Side.value_counts()[0]
    trials_right = df_session.Side.value_counts()[1]

    # Hits
    hits = df_session.Hit.sum().astype(int)
    hits_left = df_session.Hit[df_session.Side == 0].sum().astype(int)
    hits_right = df_session.Hit[df_session.Side == 1].sum().astype(int)

    # Errors
    errors = df_session.WrongLick.sum().astype(int) + df_session.Punish.sum().astype(int)
    errors_left = df_session.WrongLick[df_session.Side == 0].sum().astype(int) + df_session.Punish[
        df_session.Side == 0].sum().astype(int)
    errors_right = df_session.WrongLick[df_session.Side == 1].sum().astype(int) + df_session.Punish[
        df_session.Side == 1].sum().astype(int)

    # Performance
    performance = hits / trials
    performance_left = hits_left / trials_left
    performance_right = hits_right / trials_right

    # Responses (valid trials)
    responses = df_session.Response.sum()
    responses_left = df_session.Response[df_session.Side == 0].sum()
    responses_right = df_session.Response[df_session.Side == 1].sum()

    # Accuracy (hit rate)
    accuracy = hits / responses
    accuracy_left = hits_left / responses_left
    accuracy_right = hits_right / responses_right

    # Misses (invalid trials)
    misses = df_session.Miss.sum()
    misses_left = df_session.Miss[df_session.Side == 0].sum()
    misses_right = df_session.Miss[df_session.Side == 1].sum()

    # Miss rate
    miss_rate = misses / trials
    miss_rate_left = misses_left / trials_left
    miss_rate_right = misses_right / trials_right

    # Reward
    rewards = df_session.Reward.sum()
    rewards_left = df_session.Reward[df_session.Side == 0].sum()
    rewards_right = df_session.Reward[df_session.Side == 1].sum()

    # Water
    reward_size = 2.5  # μL
    water = rewards * reward_size
    water_left = rewards_left * reward_size
    water_right = rewards_right * reward_size

    ####################################################################################################################

    # SUMMARY TEXT

    s1 = ('Date: ' + df_session.Date.unique()[0] + ', ' +
          'Time: ' + df_session.SessionStart.unique()[0][0:-7] + ' - ' + df_session.SessionEnd.unique()[0][
                                                                         0:-7] + ', ' +
          'Subject: ' + df_session.Subject.unique()[0] + ', ' +
          'Box: ' + df_session.Board.unique()[0][4] +
          '\n')
    # [0:-7] to get rid of the floating numbers in the seconds

    s2 = ('Stage: ' + str(df_session.Stage.unique()[0]) + ', ' +
          'Substage: ' + str(df_session.Substage.unique()[0]) + ', ' +
          'Fixation: ' + str(df_session.Fixation.unique()[0]) + ', ' +
          'Timeout: ' + str(df_session.Timeout.unique()[0]) + ', ' +
          'Switch: ' + str(df_session.Switch.unique()[0]) + ', ' +
          'Motor: ' + str(df_session.Motor.unique()[0]) + ', ' +
          'CB: ' + str(df_session.CB.unique()[0]) + ', ' +
          'Progression: ' + str(df_session.Progression.unique()[0]) +
          '\n')

    # s3 = ('Total trials: ' + str(trials) + ', ' +
    #       'Performance: ' + str(round(performance * 100)) + '%' + ', ' +
    #       'Hits left:' + str(hits_left) + ' (' + str(round(performance_left * 100)) + '%)' + ', ' +
    #       'Hits right: ' + str(hits_right) + ' (' + str(round(performance_right * 100)) + '%)' +
    #       '\n')

    s3 = ('Total trials: ' + str(trials) + ' (' + str(trials_left) + ' L, ' + str(trials_right) + ' R)' + ', ' +
          'Performance: ' + str(round(performance * 100)) + '% (' + str(round(performance_left * 100)) + '% L, ' + str(
                round(performance_right * 100)) + '% R)' + ', ' +
          'Accuracy: ' + str(round(accuracy * 100)) + '% (' + str(round(accuracy_left * 100)) + '% L, ' + str(
                round(accuracy_right * 100)) + '% R)' +
          '\n')

    # s4 = ('Responses: ' + str(responses) + ', ' +
    #       'Accuracy: ' + str(round(accuracy * 100)) + '%' + ', ' +
    #       'Hits left: ' + str(hits_left) + ' (' + str(round(accuracy_left * 100)) + '%)' + ', ' +
    #       'Hits right: ' + str(hits_right) + ' (' + str(round(accuracy_right * 100)) + '%)' +
    #       '\n')

    s4 = ('Responses: ' + str(responses) + ' (' + str(responses_left) + ' L, ' + str(responses_right) + ' R)' + ', ' +
          'Hits: ' + str(hits) + ' (' + str(hits_left) + ' L, ' + str(hits_right) + ' R)' + ', ' +
          'Errors: ' + str(errors) + ' (' + str(errors_left) + ' L, ' + str(errors_right) + ' R)' +
          '\n')

    s5 = ('Misses: ' + str(misses) + ' (' + str(round(miss_rate * 100, 1)) + '%)' + ', ' +
          'Miss left: ' + str(misses_left) + ' (' + str(round(miss_rate_left * 100)) + '%)' + ', ' +
          'Miss right: ' + str(misses_right) + ' (' + str(round(miss_rate_right * 100)) + '%)' +
          '\n')

    s6 = ('Water: ' + str(water) + ' μL' + ', ' +
          'Water left: ' + str(water_left) + ' μL' + ', ' +
          'Water right: ' + str(water_right) + ' μL' + ', ' +
          'AW: ' + str(df_session.AW.unique()[0]) + ' μL' +
          '\n')

    # plt.text(0.1, 0.90, s1 + s2 + s3 + s4 + s5 + s6, fontsize=8, transform=plt.gcf().transFigure)

    ####################################################################################################################

    # fig = plt.figure()

    # PLOT 1: ACCURACY PER SIDE

    time_start_acc_side = time.time()

    # Compute accuracy rolling average
    ra_total = compute_window(df_session.Hit[df_session.Miss == 0], 20)  # All valid trials
    ra_left = compute_window(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 0)],
                             20)  # Left valid trials
    ra_right = compute_window(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 1)],
                              20)  # Right valid trials

    # Plot horizontal lines
    ax1.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
    ax1.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
    ax1.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

    # Plot accuracy rolling average
    ax1.plot(df_session.Hit[df_session.Miss == 0].index, ra_total, marker='o', ms=ms, lw=lw, color='black',
             label='Total')
    ax1.plot(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 0)].index, ra_left, marker='o', ms=ms,
             lw=lw,
             color='tab:blue',
             label='Left')
    ax1.plot(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 1)].index, ra_right, marker='o', ms=ms,
             lw=lw,
             color='tab:orange',
             label='Right')

    ax1.set_xlim([1, len(df_session)])  # 1 to not plot trial 0
    ax1.set_xticklabels([])
    ax1.set_ylabel('Acc.\n(%)')
    ax1.set_ylim([0, 1.1])
    ax1.set_yticks(list(np.arange(0, 1.1, 0.1)))
    ax1.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
    # ax1.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
    ax1.legend(loc='lower right', fontsize='xx-small', frameon=True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    # ax1.spines['right'].set_visible(False)

    # Instantiate a second axes that shares the same x-axis
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylim([0, 1.1])
    ax1_twin.set_yticks(list(np.arange(0, 1.1, 0.1)))
    ax1_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
    ax1_twin.spines['top'].set_visible(False)
    ax1_twin.spines['bottom'].set_visible(False)

    time_end_acc_side = time.time()
    runtime_acc_side = time_end_acc_side - time_start_acc_side
    print("'Plot 1: accuracy per side' took", round(runtime_acc_side, 2), 'seconds to run')

    # Plot text
    ax1.text(0, 1, s1 + s2 + s3 + s4 + s5 + s6)

    trials['y_val'] = 1
    labels = trials.trial_result.unique().tolist()
    colors = trials.color.unique().tolist()
    lines = [Line2D([0], [0], color=c, marker='o', markersize=7, markerfacecolor=c) for c in colors]
    custom_palette = sns.set_palette(sns.color_palette(colors))
    # Plot 1
    ax1.clear()
    sns.scatterplot(x=trials.TRIAL, y=trials.y_val, hue=trials.trial_result, palette=custom_palette, s=30, ax=ax1)
    ax1.set_ylabel('')
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['', '', ''])
    ax1.set_xlabel('')
    # ax1.set_xlim(0, x_max)
    ax1.legend(lines, labels, title='Trial result')
    # Plot2
    ax2.clear()
    sns.scatterplot(x=trials.TRIAL, y=trials.valid_bool, color='black', s=30, ax=ax2)
    sns.lineplot(x=trials.TRIAL, y=trials.valid_bool, color='black', ax=ax2)
    ax2.hlines(y=[0.5, 1], xmin=0, xmax=x_max, color='gray', linestyle=':', linewidth=1)
    ax2.set_ylabel('Valids (%)')
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
    ax2.set_ylim(0, 1.1)


def animate(i):
    try:
        df_session = parse(path)
    except:
        df_session = pd.DataFrame()
    if df_session.shape[0] > 0:
        try:
            plot(df_session)
        except:
            pass


fig = plt.figure()
ax1 = fig.add_subplot(4, 1, 1)
ax2 = fig.add_subplot(4, 1, 2)
ax3 = fig.add_subplot(4, 1, 3)
ax4 = fig.add_subplot(4, 3, 10)
ax5 = fig.add_subplot(4, 3, 11)
ax6 = fig.add_subplot(4, 3, 12)

path = ''

x_max = 20
interval = 2000

ani = FuncAnimation(fig, animate, interval=interval)
plt.tight_layout()
sns.despine()
plt.show()
