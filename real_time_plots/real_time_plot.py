import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # For custom legend
from matplotlib.lines import Line2D  # For custom legend
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import pandas as pd
import time
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

    # PLOT 4: HIT SCATTER PLOT

    time_start_hit = time.time()

    palette = ['tab:red', 'tab:green', 'grey']
    hue = ['Error' if i == 0 else 'Hit' if i == 1 else 'Miss' for i in df.Hit]
    hue_order = ['Error', 'Hit', 'Miss']

    if df.Stage.unique()[0] <= 3:  # No coherences, plot sides
        scatter = sns.scatterplot(x=df.index, y=df.Side, hue=hue, palette=palette,
                                  hue_order=hue_order,
                                  s=ms ** 2, ax=ax4)
        # ax4.set_ylim(-0.8, 1.8)
        # ax4.set_yticks([0, 1])
        # ax4.set_yticklabels(['L', 'R'])
        # ax4.set_ylabel('Sides')
        #
        # # Instantiate a second axes that shares the same x-axis
        # ax4_twin = ax4.twinx()
        # ax4_twin.set_ylim(-0.8, 1.8)  # Evidences
        # ax4_twin.set_yticks([0, 1])
        # ax4_twin.set_yticklabels(['L', 'R'])
        # ax4_twin.spines['top'].set_visible(False)

    else:  # Plot coherences
        scatter = sns.scatterplot(x=df.index, y=df.Evidence, hue=hue, palette=palette,
                                  hue_order=hue_order, s=ms ** 2, ax=ax4)
    #     # Plot horizontal lines
    #     ax4.axhline(0, color='tab:gray', linestyle='--')  # Evidence 0
    #     ax4.axhline(-0.5, color='tab:gray', linestyle=':')  # Evidence -0.5
    #     ax4.axhline(0.5, color='tab:gray', linestyle=':')  # Evidence 0.5
    #     ax4.set_ylim(-1.1, 1.1)  # Evidences
    #     # ax.set_ylim(0, 1)  # Coherences
    #     ax4.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    #     ax4.set_yticklabels(['L', '', '', '', '0', '', '', '', 'R'])  # Evidences
    #     # ax.set_yticklabels(['Left', '', '', '', '0.5', '', '',  '', 'Right'])  # Coherences
    #     ax4.set_ylabel('Evi.')
    #     # ax.set_ylabel('Coherence')
    #
    #     # Instantiate a second axes that shares the same x-axis
    #     ax4_twin = ax4.twinx()
    #     ax4_twin.set_ylim(-1.1, 1.1)  # Evidences
    #     ax4_twin.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    #     ax4_twin.set_yticklabels(['', '', '', '', '', '', '', '', ''])
    #     ax4_twin.spines['top'].set_visible(False)
    #
    # ax4.set_xlim([1, len(df)])  # 1 to not plot trial 0
    # ax4.set_xlabel('Trial')
    # # scatter.legend(bbox_to_anchor=(1, 1))
    # scatter.legend(loc='lower right', fontsize='xx-small', frameon=True)
    # # scatter.get_legend().remove()
    # ax4.spines['top'].set_visible(False)
    # # ax4.spines['right'].set_visible(False)

    time_end_hit = time.time()
    runtime_hit = time_end_hit - time_start_hit
    print("'Plot 4: misses' took", round(runtime_hit, 2), 'seconds to run')

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