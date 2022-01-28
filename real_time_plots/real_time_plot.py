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

    print(ax1)
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # accuracy = round(df.Hit.mean(), 2)
    # accuracy_left = round(df.loc[df.Choice == 0].Hit.mean(), 2)
    # accuracy_right = round(df.loc[df.Choice == 1].Hit.mean(), 2)
    # reward = df.loc[df.Hit == 1].shape[0]
    # water = 2.5 * reward

    if df is None:
        return

<<<<<<< HEAD
    df = df.tail(trials + running_window)  # + For computing rolling average 
=======
    df = df.tail(trials + running_window)  # + For computing rolling average
>>>>>>> efe2d551c06ea84eaf3090c23ba8ab300b4f4d57

    extra_trials = max(len(df) - trials, 0)
    x_min = df.index[0] + extra_trials
    x_max = df.index[-1]

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

    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([-0.1, 1.1])

    text = box + ': ' + path
    ax1.text(0, 1.1, text, transform=ax1.transAxes)
    # text2 = 'Acc: ' + str(accuracy)
    # ax1.text(0, 0, text2, transform=ax2.transAxes)
    ####################################################################################################################

    # PLOT 2: SOUND ERRORS

    scatter = sns.scatterplot(x=df.Trial, y=df.Message - 1, ax=ax2, color='purple')
    scatter = sns.scatterplot(x=df.Trial, y=df.Sound, ax=ax2, color='red')
    scatter = sns.scatterplot(x=df.Trial, y=df.FilesMatch, ax=ax2, color='pink')

    ax2.set_xlim([x_min, x_max])
    ax2.set_ylabel('Sound errors')
    scatter.set_ylim([-0.1, 0.1])
    ####################################################################################################################

    # PLOT 3: MISSES

    # Compute accuracy rolling average
    ra_total_miss = compute_window(df.Miss, 20)  # All valid trials
    ra_left_miss = compute_window(df.Miss[df.Side == 0], 20)  # Left valid trials
    ra_right_miss = compute_window(df.Miss[df.Side == 1], 20)  # Right valid trials

    # Plot horizontal lines
    ax3.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
    ax3.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
    ax3.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

    # Plot misses rolling average
    ax3.plot(df.index, ra_total_miss, marker='o', ms=ms, lw=lw, color='black', label='Total')
    ax3.plot(df[df.Side == 0].index, ra_left_miss, marker='o', ms=ms, lw=lw, color='tab:blue',
             label='Left')
    ax3.plot(df[df.Side == 1].index, ra_right_miss, marker='o', ms=ms, lw=lw, color='tab:orange',
             label='Right')

    ax3.set_xlabel('Trial')
    ax3.set_xlim([x_min, x_max])
    ax3.set_ylabel('Misses')
    ax3.set_ylim([-0.1, 1.1])
    ####################################################################################################################

    # PLOT 4: HIT SCATTER PLOT

    palette = ['tab:red', 'tab:green', 'grey']
    hue = ['Error' if i == 0 else 'Hit' if i == 1 else 'Miss' for i in df.Hit]
    hue_order = ['Error', 'Hit', 'Miss']

    if df.Stage.unique()[0] <= 3:  # No coherences, plot sides
        scatter = sns.scatterplot(x=df.Trial, y=df.Side, hue=hue, palette=palette,
                                  hue_order=hue_order,
                                  s=ms ** 2, ax=ax4)
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['L', 'R'])

    else:  # Plot coherences
        scatter = sns.scatterplot(x=df.Trial, y=df.Evidence, hue=hue, palette=palette,
                                  hue_order=hue_order, s=ms ** 2, ax=ax4)

    scatter.get_legend().remove()
    ax4.set_xlim([x_min, x_max])


