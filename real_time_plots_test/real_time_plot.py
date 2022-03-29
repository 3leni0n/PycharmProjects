import seaborn as sns
import numpy as np
import pandas as pd
import time
# from my_fun.my_fun import *


def real_time_plot(df, box, path, ax1, ax2, ax3, ax4, trials):

    ####################################################################################################################

    # Plotting style
    ms = 3  # (mpl default for plt.plot=6 and for plt.scatter=6**2)
    lw = 1.5  # mpl default
    running_window = 20

    ####################################################################################################################

    def compute_window(data, runningwindow):
        """
        Computes a rolling average with a length of runningwindow samples.
        """
        performance = []
        for i in range(len(data)):
            if i < runningwindow:
                performance.append(round(np.mean(data[0:i]), 2))
            else:
                performance.append(round(np.mean(data[i - runningwindow:i]), 2))
        return performance

    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    if df is None:
        return

    df = df.tail(trials + running_window)  # For computing rolling average
    print(df)
    print('\n')

    extra_trials = max(len(df) - trials, 0)
    print(extra_trials)

    x_min = df.index[0] + extra_trials
    x_max = df.index[-1]

    y_min = df.index[0] + extra_trials  # For the lick raster
    y_max = df.index[-1]  # For the lick raster

    ####################################################################################################################

    # PLOT 1: ACCURACY PER SIDE

    # Compute accuracy rolling average
    ra_total = compute_window(df.Hit[df.Miss == 0], 20)  # All valid trials
    ra_left = compute_window(df.Hit[(df.Miss == 0) & (df.Side == 0)], 20)  # Left valid trials
    ra_right = compute_window(df.Hit[(df.Miss == 0) & (df.Side == 1)], 20)  # Right valid trials


    # Plot horizontal lines
    ax1.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
    ax1.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
    ax1.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

    # Plot accuracy rolling average
    ax1.plot(df[df.Miss == 0].index, ra_total, marker='o', ms=ms, lw=lw, color='black', label='Total')
    ax1.plot(df[(df.Miss == 0) & (df.Side == 0)].index, ra_left, marker='o', ms=ms, lw=lw, color='tab:blue',
             label='Left')
    ax1.plot(df[(df.Miss == 0) & (df.Side == 1)].index, ra_right, marker='o', ms=ms, lw=lw, color='tab:orange',
             label='Right')

    scatter = sns.scatterplot(x=df.Trial, y=df.Message - 1, ax=ax1, color='purple')
    scatter = sns.scatterplot(x=df.Trial, y=df.Sound, ax=ax1, color='red')
    scatter = sns.scatterplot(x=df.Trial, y=df.FilesMatch, ax=ax1, color='pink')

    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([-0.1, 1.1])

    text = box + ': ' + path
    ax1.text(0, 1.1, text, transform=ax1.transAxes)

    ####################################################################################################################

    # PLOT 2: MISSES

    # Compute accuracy rolling average
    ra_total_miss = compute_window(df.Miss, 20)  # All valid trials
    ra_left_miss = compute_window(df.Miss[df.Side == 0], 20)  # Left valid trials
    ra_right_miss = compute_window(df.Miss[df.Side == 1], 20)  # Right valid trials

    # Plot horizontal lines
    ax2.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
    ax2.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
    ax2.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

    # Plot misses rolling average
    ax2.plot(df.index, ra_total_miss, marker='o', ms=ms, lw=lw, color='black', label='Total')
    ax2.plot(df[df.Side == 0].index, ra_left_miss, marker='o', ms=ms, lw=lw, color='tab:blue',
             label='Left')
    ax2.plot(df[df.Side == 1].index, ra_right_miss, marker='o', ms=ms, lw=lw, color='tab:orange',
             label='Right')

    scatter = sns.scatterplot(x=df.Trial, y=df.Message - 1, ax=ax2, color='purple')
    scatter = sns.scatterplot(x=df.Trial, y=df.Sound, ax=ax2, color='red')
    scatter = sns.scatterplot(x=df.Trial, y=df.FilesMatch, ax=ax2, color='pink')

    ax2.set_xlabel('Trial')
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylabel('Misses')
    ax2.set_ylim([-0.1, 1.1])

    ####################################################################################################################

    # PLOT 3: HIT SCATTER PLOT

    palette = ['tab:red', 'tab:green', 'grey']
    hue = ['Error' if i == 0 else 'Hit' if i == 1 else 'Miss' for i in df.Hit]
    hue_order = ['Error', 'Hit', 'Miss']

    if df.Stage.unique()[0] <= 3:  # No coherences, plot sides
        scatter = sns.scatterplot(x=df.Trial, y=df.Side, hue=hue, palette=palette,
                                  hue_order=hue_order,
                                  s=ms ** 2, ax=ax3)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['L', 'R'])

    else:  # Plot coherences
        scatter = sns.scatterplot(x=df.Trial, y=df.ILD, hue=hue, palette=palette,
                                  hue_order=hue_order, s=ms ** 2, ax=ax3)
        ax3.set_yticks(df.ILD.unique())
        # ax3.set_yscale('log')

    scatter = sns.scatterplot(x=df.Trial, y=df.Message - 1, ax=ax3, color='purple')
    scatter = sns.scatterplot(x=df.Trial, y=df.Sound, ax=ax3, color='red')
    scatter = sns.scatterplot(x=df.Trial, y=df.FilesMatch, ax=ax3, color='pink')

    scatter.get_legend().remove()
    ax3.set_xlim([x_min, x_max])

    ax3.set_xlabel('Trial')
    ax3.set_xlim([x_min, x_max])
    ax3.set_ylabel('ILD')

    ####################################################################################################################

    # PLOT 4: PERISTIMULUS LICK RASTER

    last = df.Trial.iloc[-1]

    if last > trials:
        first = last - trials
    else:
        first = 0

    for j in first:last:

    for j in range(len(df)):  # n trials

        k = j + len(df) - trials

        if df.Side.iloc[j] == 0:
            stim_color = 'tab:blue'
        elif df.Side.iloc[j] == 1:
            stim_color = 'tab:orange'

        # Plot stimulus length
        ax4.barh(df.index.values[j], df.StimLen.iloc[j], left=df.StimStart.iloc[j] - df.StimStart.iloc[j],
                 color=stim_color, alpha=1, label='Stim', zorder=1)
        # Need to specify zorder otherwise response window is plotted under stimulus length and can't be seen

        # Define response window color according to trial outcome
        if df.WrongLick[j] == 1.0:
            resp_win_color = 'tab:pink'
        elif df.Hit[j] == 0.0:
            resp_win_color = 'tab:red'
        elif df.Hit[j] == 1.0:
            resp_win_color = 'tab:green'
        elif np.isnan(df.Hit[j]):
            resp_win_color = 'tab:gray'

        # Plot response window length
        ax4.barh(df.index.values[j], df.RespWinLen[j], left=df.RespWinStart[j] - df.StimStart[j], color=resp_win_color,
                 zorder=2)

        # Left licks
        for i in range(len(df.Port1In[j])):  # n licks

            # If licks are after stimulus onset, draw markeredgecolor so it can be seen over stimulus length barh
            if df.Port1In[j][i] - \
                    df.StimStart[j] > \
                    df.StimStart[j] - \
                    df.StimStart[j]:
                ms = 2
                mec = 'k'
                mew = 0.1
            else:
                ms = 1
                mec = 'tab:blue'
                mew = None

            if not df.Port1In[j]:
                pass
            else:
                ax4.plot(df.Port1In[j][i] -
                        df.StimStart[j],
                        df.index[j], marker='o', ms=200 / len(df.Side == 0),
                        mec=mec, mew=mew, color='tab:blue', zorder=3)
                # markersize=200 / len(df.Side == 0))
                # markersize = ax.containers[1][0].get_height()

        # Right licks
        for i in range(len(df.Port2In[j])):  # n licks

            # If licks are after stimulus onset, draw markeredgecolor so it can be seen over stimulus length barh
            if df.Port2In[j][i] - \
                    df.StimStart[j] > \
                    df.StimStart[j] - \
                    df.StimStart[j]:
                ms = 2
                mec = 'k'
                mew = 0.1
            else:
                ms = 1
                mec = 'tab:orange'
                mew = None

            if not df.Port2In[j]:
                pass
            else:
                ax4.plot(df.Port2In[j][i] -
                        df.StimStart[j],
                        df.Port2In.index[j], marker='o', ms=200 / len(df.Side == 0),
                        mec=mec, mew=mew, color='tab:orange', zorder=3)
                # markersize=200 / len(df.Side == 1))
                # markersize = ax.containers[1][0].get_height()

    ax4.set_xlabel('Time (s) from stim. onset')
    ax4.set_ylabel('Trial')
    # ax4.set_ylim([max(len(df) - trials, 0), len(df)])
    # print(y_min, y_max)

    # Plot text
    accuracy = round(df.Hit.mean(), 2)
    accuracy_left = round(df.loc[df.Side == 0].Hit.mean(), 2)
    accuracy_right = round(df.loc[df.Side == 1].Hit.mean(), 2)
    reward = df.loc[df.Hit == 1].shape[0]
    water = 2.5 * reward

    ax4.text(1, 1,
             "Acc=" + str(accuracy) + "\n" +  # Accuracy
             "Acc. left=" + str(accuracy_left) + "\n" +  # Accuracy left
             "Acc. right=" + str(accuracy_right) + "\n" +  # Accuracy right
             "Water=" + str(water) + "\n",  # Water
             transform=ax4.transAxes, color='k', va='top', ha='left', fontsize='small')
