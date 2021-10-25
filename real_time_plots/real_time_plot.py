import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # For custom legend
from matplotlib.lines import Line2D  # For custom legend
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import pandas as pd
# from my_fun.my_fun import *


def real_time_plot(df, path, ax1, ax2, ax3, ax4, trials):

    ####################################################################################################################

    # Plotting style
    ms = 3  # (mpl default for plt.plot=6 and for plt.scatter=6**2)
    lw = 1.5  # mpl default

    ####################################################################################################################

    def compute_window(data, runningwindow):
        """
        Computes a rolling average with a length of runningwindow samples.
        """
        performance = []
        for i in range(len(data)):
            if i < runningwindow:
                performance.append(round(np.mean(data[0:i + 1]), 2))
            else:
                performance.append(round(np.mean(data[i - runningwindow:i]), 2))
        return performance

    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # accuracy = round(df_session.Hit.mean(), 2)
    #
    # accuracy_left = round(df_session.loc[df_session.Choice == 0].Hit.mean(), 2)
    # accuracy_right = round(df_session.loc[df_session.Choice == 1].Hit.mean(), 2)
    # rewards = df_session.loc[df_session.Hit == 1].shape[0]
    #
    # water = 2.5 * rewards

    df = df.tail(trials)

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
    ax1.plot(df.Hit[df.Miss == 0].index, ra_total, marker='o', ms=ms, lw=lw, color='black', label='Total')
    ax1.plot(df.Hit[(df.Miss == 0) & (df.Side == 0)].index, ra_left, marker='o', ms=ms, lw=lw, color='tab:blue',
             label='Left')
    ax1.plot(df.Hit[(df.Miss == 0) & (df.Side == 1)].index, ra_right, marker='o', ms=ms, lw=lw, color='tab:orange',
             label='Right')

    ####################################################################################################################

    # text1 = ('Task: ' + path + '\n')
    # text2 = ('Accuracy: ' + str(accuracy) + '  Accuracy left: ' + str(accuracy_left) + '  Accuracy right: ' + str(accuracy_right) + '\n')
    # text3 = ('Water: ' +  str(water))


    # palette = {0: "black",
    #            1: "green",
    #            0.5: "gray"}


    # df = df.fillna(0.5)

    # ax2.text(0, 0, text1 + text2 + text3)
    #
    #
    # sns.scatterplot(x=df_session.Trial, y=df_session.Choice, hue=df_session.Hit, palette = palette, s=100, ax=ax1)

    # ax2.set_axis_off()
    #
    # ax1.legend([], [], frameon=False)
    #
    # sns.despine()