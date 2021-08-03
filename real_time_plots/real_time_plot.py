import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # For custom legend
from matplotlib.lines import Line2D  # For custom legend
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import pandas as pd

from my_fun.my_fun import compute_window


def real_time_plot(df_session, ax1, ax2, ax3):

    # Plotting style
    ms = 3  # (mpl default for plt.plot=6 and for plt.scatter=6**2)
    lw = 1.5  # mpl default

    ax1.clear()
    ax2.clear()
    ax3.clear()

########################################################################################################################

    # PLOT 1: ACCURACY PER SIDE

    # Compute accuracy rolling average
    ra_total = compute_window(df_session.Hit[df_session.Miss == 0], 20)  # All valid trials
    ra_left = compute_window(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 0)],
                             20)  # Left valid trials
    ra_right = compute_window(df_session.Hit[(df_session.Miss == 0) & (df_session.Side == 1)],
                              20)  # Right valid trials

    # # Prepares the grid for the plots
    # ax1 = plt.subplot2grid((16, 4), (0, 0), rowspan=4, colspan=4)
    # # ax1 = plt.subplot2grid((4, 1), (0, 0))

    # Prepares the grid for the plots
    # if df_session.Stage.unique()[0] == 4:
    #     ax1 = plt.subplot2grid((16, 4), (0, 0), rowspan=3, colspan=4)
    # else:
    #     ax1 = plt.subplot2grid((16, 4), (0, 0), rowspan=4, colspan=4)
    # ax1 = plt.subplot2grid((4, 1), (0, 0))

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
    # ax1.legend(loc='lower right', fontsize='xx-small', frameon=True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    # ax1.spines['right'].set_visible(False)

    # # Instantiate a second axes that shares the same x-axis
    # ax1_twin = ax1.twinx()
    # ax1_twin.set_ylim([0, 1.1])
    # ax1_twin.set_yticks(list(np.arange(0, 1.1, 0.1)))
    # ax1_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
    # ax1_twin.spines['top'].set_visible(False)
    # ax1_twin.spines['bottom'].set_visible(False)

########################################################################################################################

    # PLOT 4: HIT SCATTER PLOT

    # # Prepares the grid for the plots
    # if df_session.Stage.unique()[0] == 4:
    #     ax4 = plt.subplot2grid((16, 4), (7, 0), rowspan=3, colspan=4)
    # else:
    #     ax4 = plt.subplot2grid((16, 4), (12, 0), rowspan=4, colspan=4)
    # # ax4 = plt.subplot2grid((4, 1), (3, 0))

    palette = ['tab:red', 'tab:green', 'grey']
    hue = ['Error' if i == 0 else 'Hit' if i == 1 else 'Miss' for i in df_session.Hit]
    hue_order = ['Error', 'Hit', 'Miss']

    if df_session.Stage.unique()[0] <= 3:  # No coherences, plot sides
        scatter = sns.scatterplot(x=df_session.index, y=df_session.Side, hue=hue, palette=palette, hue_order=hue_order,
                                  s=ms ** 2, ax=ax2)
        ax2.set_ylim(-0.8, 1.8)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['L', 'R'])
        ax2.set_ylabel('Sides')

        # Instantiate a second axes that shares the same x-axis
        # ax2_twin = ax2.twinx()
        # ax2_twin.set_ylim(-0.8, 1.8)  # Evidences
        # ax2_twin.set_yticks([0, 1])
        # ax2_twin.set_yticklabels(['L', 'R'])
        # ax2_twin.spines['top'].set_visible(False)

    else:  # Plot coherences
        scatter = sns.scatterplot(x=df_session.index, y=df_session.Evidence, hue=hue, palette=palette,
                                  hue_order=hue_order, s=ms ** 2, ax=ax2)
        # Plot horizontal lines
        ax2.axhline(0, color='tab:gray', linestyle='--')  # Evidence 0
        ax2.axhline(-0.5, color='tab:gray', linestyle=':')  # Evidence -0.5
        ax2.axhline(0.5, color='tab:gray', linestyle=':')  # Evidence 0.5
        ax2.set_ylim(-1.1, 1.1)  # Evidences
        # ax.set_ylim(0, 1)  # Coherences
        ax2.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
        ax2.set_yticklabels(['L', '', '', '', '0', '', '', '', 'R'])  # Evidences
        # ax.set_yticklabels(['Left', '', '', '', '0.5', '', '',  '', 'Right'])  # Coherences
        ax2.set_ylabel('Evi.')
        # ax.set_ylabel('Coherence')

        # Instantiate a second axes that shares the same x-axis
        # ax2_twin = ax2.twinx()
        # ax2_twin.set_ylim(-1.1, 1.1)  # Evidences
        # ax2_twin.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
        # ax2_twin.set_yticklabels(['', '', '', '', '', '', '', '', ''])
        # ax2_twin.spines['top'].set_visible(False)

    ax2.set_xlim([1, len(df_session)])  # 1 to not plot trial 0
    ax2.set_xlabel('Trial')
    # scatter.legend(bbox_to_anchor=(1, 1))
    # scatter.legend(loc='lower right', fontsize='xx-small', frameon=True)
    # scatter.get_legend().remove()
    ax2.spines['top'].set_visible(False)
    # ax4.spines['right'].set_visible(False)



    # sns.scatterplot(x=df_session.Reward, y=df_session.Response, color='black', s=100, ax=ax1)
    # sns.scatterplot(x=df_session.Response, y=df_session.Reward, color='black', s=100, ax=ax2)
    # sns.scatterplot(x=df_session.Response, y=df_session.Reward, color='black', s=100, ax=ax3)

    sns.despine()
