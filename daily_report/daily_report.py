# To do:
# Check using GridSpec instead of plt.subplot2grid as suggested by matplotlib doc
# (https://matplotlib.org/stable/gallery/userdemo/demo_gridspec01.html)
# Create and keep a different axis for every subplot (ax1, ax2, ax3, etc) instead of overwriting a single axis 'ax'
# Psychometric curves for repeating and alternating evidence
# Clean, dense to read!
# Set timers per plot to know which one is causing the code to run slow af
# Finish fixing bin_size --> weights
# In motor 2 the stimulus is only plotted in miss and error trials (but not incorrect ones)

# Make code detect OS and fill the destiny path automatically
# Check if Filename and Filename2 match: df.Filename.equals(df.Filename2), np.unique(np.equal(df.Filename, df.Filename2))
# np.where(df.Filename != df.Filename2)
# Fix the len os stim bar in error and misses trials

"""
# Tiffany's comments:

1. Markers are too big in the trend plots (miss, accuracy, repeat).DONE
2. You need to have better resolution of the Accuracy plot than of the Miss plot, so they shouldn't have the same size.
   Same for Repeat/Alternate, especially in this early stages. DONE
3. I see that you are plotting the psycometric already. I would have removed it and only display it when necessary,
   specially in order to increase the size of other more important plots. You wanna have all the information needed at a
   glance. DONE
4. Raster plot.
    a. Why are you plotting the stimulus? You already segregated by stimulus in two rasters, left and right. This is
    only obscuring the view and adding confusion. If you wanna mark the sound duration, much lighter neutral color. DONE
    b. Something still looks funny in the licks for me. Too much aligned to the left. Also, why do licks suddenly stop
    on the right? DONE
    c. Also, the display from -3 to 2 is not helpful. You wanna now what they do after reward as well, this -3 is not
    useful and it will become even less useful in the future. Put something more like from -1 until end of trial (6?)
    d. The psth for all the fist lick should have a much narrower window, you are mainly insterested in only the first 1
     or 2s tops. Then you should also change the bin size to add more resolution. DONE
    3. Also, but this is a more personal opinion, I wouldn't use bars to mark reward, punish and miss but rather an icon.
    It's a single event, reward doesn't last the duration of the bar, for instance. Also it happens after a lick, not before.
"""

########################################################################################################################

import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # For custom legend
from matplotlib.lines import Line2D  # For custom legend
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import pandas as pd
import time
import os
import slack

from my_fun.my_fun import *  # Or from daily_report.daily_report import daily_report
from parse.parse import *  # Or from parse.parse import parse


########################################################################################################################

# Define function
def daily_report(path, send_slack=False):
    # Register time
    time_start_total = time.time()

    ####################################################################################################################

    # Import session to be parsed
    df = parse(path)

    ####################################################################################################################

    # Select the folder and create it if it doesn't exist
    experiment = df.Experiment.unique()[0]  # Batch ID
    folder = '/home/alexis/Documentos/daily reports/' + experiment + '/'
    # folder = '/home/setup2/Documents/daily reports/' + experiment + '/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    os.chdir(folder)
    setup = df.Setup.unique()[0]  # Animal ID
    folder = folder + setup
    if not os.path.exists(folder):
        os.mkdir(folder)
    os.chdir(folder)

    ####################################################################################################################

    # Plotting style
    ms = 3  # (mpl default for plt.plot=6 and for plt.scatter=6**2)
    lw = 1.5  # mpl default
    # width = lw  # Arrow
    # head_width = 3 * width
    # head_length = 0.1 # 1.5 * head_width

    ####################################################################################################################

    # SUMMARY VARIABLES

    # Trials
    trials = len(df)
    trials_left = df.Side.value_counts()[0]
    trials_right = df.Side.value_counts()[1]

    # Hits
    hits = df.Hit.sum().astype(int)
    hits_left = df.Hit[df.Side == 0].sum().astype(int)
    hits_right = df.Hit[df.Side == 1].sum().astype(int)

    # Errors
    errors = df.WrongLick.sum().astype(int) + df.Punish.sum().astype(int)
    errors_left = df.WrongLick[df.Side == 0].sum().astype(int) + df.Punish[
        df.Side == 0].sum().astype(int)
    errors_right = df.WrongLick[df.Side == 1].sum().astype(int) + df.Punish[
        df.Side == 1].sum().astype(int)

    # Performance
    performance = hits / trials
    performance_left = hits_left / trials_left
    performance_right = hits_right / trials_right

    # Responses (valid trials)
    responses = df.Response.sum()
    responses_left = df.Response[df.Side == 0].sum()
    responses_right = df.Response[df.Side == 1].sum()

    # Accuracy (hit rate)
    accuracy = hits / responses
    accuracy_left = hits_left / responses_left
    accuracy_right = hits_right / responses_right

    # Misses (invalid trials)
    misses = df.Miss.sum()
    misses_left = df.Miss[df.Side == 0].sum()
    misses_right = df.Miss[df.Side == 1].sum()

    # Miss rate
    miss_rate = misses / trials
    miss_rate_left = misses_left / trials_left
    miss_rate_right = misses_right / trials_right

    # Reward
    rewards = df.Reward.sum()
    rewards_left = df.Reward[df.Side == 0].sum()
    rewards_right = df.Reward[df.Side == 1].sum()

    # Water
    reward_size = 2.5  # μL
    water = rewards * reward_size
    water_left = rewards_left * reward_size
    water_right = rewards_right * reward_size

    # Sound
    sounds_mismatch = len(np.where(df.Filename != df.Filename2)[0])
    no_sound = len(np.where(df.Sound == 0)[0])
    message_count = len(np.where(df.Message != 'nan')[0])

    ####################################################################################################################

    with PdfPages(df.Session.unique()[0]) as pdf:

        # PAGE 1

        fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches portrait
        # fig = plt.figure(figsize=(11.69, 8.27))  # A4 size in inches landscape

        ################################################################################################################

        # SUMMARY TEXT

        s1 = ('Date: ' + df.Date.unique()[0] + ', ' +
              'Time: ' + df.SessionStart.unique()[0][0:-7] + ' - ' + df.SessionEnd.unique()[0][0:-7] + ', ' +
              # [0:-7] to get rid of the floating numbers in the seconds
              'Subject: ' + df.Subject.unique()[0] + ', ' +
              'Box: ' + df.Board.unique()[0][4] + ', ' +
              '\n')

        s2 = ('Stage: ' + str(df.Stage.unique()[0]) + ', ' +
              'Substage(s): ' + str(df.Substage.sort_values().unique()[0]) + '-' +
              str(df.Substage.sort_values().unique()[-1]) + ', ' +
              'Fixation: ' + str(df.Fixation.unique()[0]) + ', ' +
              'Timeout: ' + str(df.Timeout.unique()[0]) + ', ' +
              'Switch: ' + str(df.Switch.unique()[0]) + ', ' +
              'Motor: ' + str(df.Motor.unique()[0]) + ', ' +
              'CB: ' + str(df.CB.unique()[0]) + ', ' +
              'Progression: ' + str(df.Progression.unique()[0]) +
              '\n')

        s3 = ('Trials: ' + str(trials) + ' (' + str(trials_left) + ' L, ' + str(trials_right) + ' R)' + ', ' +
              'Performance: ' + str(int(round(performance * 100))) + '% (' + str(int(round(performance_left * 100))) +
              '% L, ' + str(int(round(performance_right * 100))) + '% R)' + ', ' +
              'Accuracy: ' + str(int(round(accuracy * 100))) + '% (' + str(int(round(accuracy_left * 100))) + '% L, ' +
              str(int(round(accuracy_right * 100))) + '% R)' +
              '\n')

        s4 = ('Responses: ' + str(responses) + ' (' + str(responses_left) + ' L, ' + str(
            responses_right) + ' R)' + ', ' +
              'Hits: ' + str(hits) + ' (' + str(hits_left) + ' L, ' + str(hits_right) + ' R)' + ', ' +
              'Errors: ' + str(errors) + ' (' + str(errors_left) + ' L, ' + str(errors_right) + ' R)' + ', ' +
              'Misses: ' + str(misses) + ' (' + str(int(round(miss_rate * 100, 1))) + '%)' +
              '\n')

        s5 = ('Miss left: ' + str(misses_left) + ' (' + str(int(round(miss_rate_left * 100))) + '%)' + ', ' +
              'Miss right: ' + str(misses_right) + ' (' + str(int(round(miss_rate_right * 100))) + '%)' + ', ' +
              'Sounds mismatch: ' + str(sounds_mismatch) + ' (' + str(
                    round((sounds_mismatch / trials) * 100, 1)) + '%)' + ', ' +
              'No sound: ' + str(no_sound) + ' (' + str(round((no_sound / trials) * 100, 1)) + '%)' +
              '\n')

        s6 = ('Water: ' + str(water) + ' μL' + ', ' +
              'Water left: ' + str(water_left) + ' μL' + ', ' +
              'Water right: ' + str(water_right) + ' μL' + ', ' +
              'AW: ' + str(df.AW.unique()[0]) + ' μL' +
              '\n')

        # plt.text(0.1, 0.90, s1 + s2 + s3 + s4 + s5 + s6, fontsize=8, transform=plt.gcf().transFigure)

        ################################################################################################################

        # fig = plt.figure()

        change_substage = df.Substage.diff()  # Find trials in which substage changes
        change_substage = change_substage[change_substage != 0].dropna()  # Omit 0s and drop first nan

        # PLOT 1: ACCURACY PER SIDE

        time_start_acc_side = time.time()

        # Compute accuracy rolling average
        ra_total = compute_window(df.Hit[df.Miss == 0], 20)  # All valid trials
        ra_left = compute_window(df.Hit[(df.Miss == 0) & (df.Side == 0)],
                                 20)  # Left valid trials
        ra_right = compute_window(df.Hit[(df.Miss == 0) & (df.Side == 1)],
                                  20)  # Right valid trials

        # # Prepares the grid for the plots
        # ax1 = plt.subplot2grid((16, 4), (0, 0), rowspan=4, colspan=4)
        # # ax1 = plt.subplot2grid((4, 1), (0, 0))

        # Prepares the grid for the plots
        if df.Stage.unique()[0] == 4:
            ax1 = plt.subplot2grid((16, 4), (0, 0), rowspan=2, colspan=4)
        else:
            ax1 = plt.subplot2grid((16, 4), (0, 0), rowspan=4, colspan=4)
        # ax1 = plt.subplot2grid((4, 1), (0, 0))

        # Plot horizontal lines
        ax1.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
        ax1.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
        ax1.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

        # Plot accuracy rolling average
        ax1.plot(df.Hit[df.Miss == 0].index, ra_total, marker='o', ms=ms, lw=lw, color='black',
                 label='Total')
        ax1.plot(df.Hit[(df.Miss == 0) & (df.Side == 0)].index, ra_left, marker='o', ms=ms,
                 lw=lw,
                 color='tab:blue',
                 label='Left')
        ax1.plot(df.Hit[(df.Miss == 0) & (df.Side == 1)].index, ra_right, marker='o', ms=ms,
                 lw=lw,
                 color='tab:orange',
                 label='Right')

        if df.Progression.unique()[0] == 1:
            for i in range(len(change_substage.index)):
                if change_substage[change_substage.index[i]] == 1:
                    # ax1.annotate(s='', xy=(change_substage.index[i], 1), xytext=(change_substage.index[i], 0),
                    #              arrowprops=dict(arrowstyle='->', color='green'))
                    ax1.plot(change_substage.index[i], 0.1, marker='^', ms=ms, lw=lw, color='tab:green')
                    ax1.annotate(s=str(df.Substage[change_substage.index[i]]),
                                 xy=(change_substage.index[i], 0.1), xytext=(change_substage.index[i], 0.2),
                                 color='tab:green', ha='center')
                elif change_substage[change_substage.index[i]] == -1:
                    # ax1.annotate(s='', xy=(change_substage.index[i], 1), xytext=(change_substage.index[i], 0),
                    #              arrowprops=dict(arrowstyle='<-', color='red'))
                    ax1.plot(change_substage.index[i], 0.1, marker='v', ms=ms, lw=lw, color='tab:red')
                    ax1.annotate(s=str(df.Substage[change_substage.index[i]]),
                                 xy=(change_substage.index[i], 0.1), xytext=(change_substage.index[i], 0.2),
                                 color='tab:red', ha='center')

        ax1.set_xlim([1, len(df)])  # 1 to not plot trial 0
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

        ################################################################################################################

        # PLOT 2: REPEATING VS ALTERNATING ACCURACY

        time_start_acc_repalt = time.time()

        # Compute accuracy rolling average for repeating vs alternating trials
        ra_rep = compute_window(df.Hit[(df.Miss == 0) & (df.RepTrial == 1)], 20)
        ra_alt = compute_window(df.Hit[(df.Miss == 0) & (df.RepTrial == 0)], 20)

        # Prepares the grid for the plots
        if df.Stage.unique()[0] == 4:
            ax2 = plt.subplot2grid((16, 4), (2, 0), rowspan=2, colspan=4)
        else:
            ax2 = plt.subplot2grid((16, 4), (4, 0), rowspan=4, colspan=4)
        # ax2 = plt.subplot2grid((4, 1), (1, 0))

        # Plot horizontal lines
        ax2.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
        ax2.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
        ax2.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

        # Plot accuracy rolling average
        ax2.plot(df.Hit[(df.Miss == 0) & (df.RepTrial == 1)].index, ra_rep, marker='o', ms=ms,
                 lw=lw, color='tab:brown', label='Rep')
        ax2.plot(df.Hit[(df.Miss == 0) & (df.RepTrial == 0)].index, ra_alt, marker='o', ms=ms,
                 lw=lw, color='tab:purple', label='Alt')

        if df.Progression.unique()[0] == 1:
            for i in range(len(change_substage.index)):
                if change_substage[change_substage.index[i]] == 1:
                    # ax2.annotate(s='', xy=(change_substage.index[i], 1), xytext=(change_substage.index[i], 0),
                    #              arrowprops=dict(arrowstyle='->', color='green'))
                    ax2.plot(change_substage.index[i], 0.1, marker='^', ms=ms, lw=lw, color='tab:green')
                    ax2.annotate(s=str(df.Substage[change_substage.index[i]]),
                                 xy=(change_substage.index[i], 0.1), xytext=(change_substage.index[i], 0.2),
                                 color='tab:green', ha='center')
                elif change_substage[change_substage.index[i]] == -1:
                    # ax2.annotate(s='', xy=(change_substage.index[i], 1), xytext=(change_substage.index[i], 0),
                    #              arrowprops=dict(arrowstyle='<-', color='red'))
                    ax2.plot(change_substage.index[i], 0.1, marker='v', ms=ms, lw=lw, color='tab:red')
                    ax2.annotate(s=str(df.Substage[change_substage.index[i]]),
                                 xy=(change_substage.index[i], 0.1), xytext=(change_substage.index[i], 0.2),
                                 color='tab:red', ha='center')

        ax2.set_xlim([1, len(df)])  # 1 to not plot trial 0
        ax2.set_xticklabels([])
        ax2.set_ylabel('Acc.\n(%)')
        ax2.set_ylim([0, 1.1])
        ax2.set_yticks(list(np.arange(0, 1.1, 0.1)))
        ax2.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        # ax2.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        ax2.legend(loc='lower right', fontsize='xx-small', frameon=True)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        # ax2.spines['right'].set_visible(False)

        # Instantiate a second axes that shares the same x-axis
        ax2_twin = ax2.twinx()
        ax2_twin.set_ylim([0, 1.1])
        ax2_twin.set_yticks(list(np.arange(0, 1.1, 0.1)))
        ax2_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        ax2_twin.spines['top'].set_visible(False)
        ax2_twin.spines['bottom'].set_visible(False)

        time_end_acc_repalt = time.time()
        runtime_acc_repalt = time_end_acc_repalt - time_start_acc_repalt
        print("'Plot 2: accuracy repeating vs alternating' took", round(runtime_acc_repalt, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 3: MISSES

        time_start_miss = time.time()

        # Compute accuracy rolling average
        ra_total_miss = compute_window(df.Miss, 20)  # All valid trials
        ra_left_miss = compute_window(df.Miss[df.Side == 0], 20)  # Left valid trials
        ra_right_miss = compute_window(df.Miss[df.Side == 1], 20)  # Right valid trials

        # Prepares the grid for the plots
        if df.Stage.unique()[0] == 4:
            ax3 = plt.subplot2grid((16, 4), (4, 0), rowspan=2, colspan=4)
        else:
            ax3 = plt.subplot2grid((16, 4), (8, 0), rowspan=4, colspan=4)
        # ax3 = plt.subplot2grid((4, 1), (2, 0))

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

        if df.Progression.unique()[0] == 1:
            for i in range(len(change_substage.index)):
                if change_substage[change_substage.index[i]] == 1:
                    # ax3.annotate(s='', xy=(change_substage.index[i], 1), xytext=(change_substage.index[i], 0),
                    #              arrowprops=dict(arrowstyle='->', color='green'))
                    ax3.plot(change_substage.index[i], 0.1, marker='^', ms=ms, lw=lw, color='tab:green')
                    ax3.annotate(s=str(df.Substage[change_substage.index[i]]),
                                 xy=(change_substage.index[i], 0.1), xytext=(change_substage.index[i], 0.2),
                                 color='tab:green', ha='center')
                elif change_substage[change_substage.index[i]] == -1:
                    # ax3.annotate(s='', xy=(change_substage.index[i], 1), xytext=(change_substage.index[i], 0),
                    #              arrowprops=dict(arrowstyle='<-', color='red'))
                    ax3.plot(change_substage.index[i], 0.1, marker='v', ms=ms, lw=lw, color='tab:red')
                    ax3.annotate(s=str(df.Substage[change_substage.index[i]]),
                                 xy=(change_substage.index[i], 0.1), xytext=(change_substage.index[i], 0.2),
                                 color='tab:red', ha='center')

        ax3.set_xlim([1, len(df)])  # 1 to not plot trial 0
        ax3.set_xticklabels([])
        ax3.set_ylim([0, 1.1])
        ax3.set_ylabel('Miss\n(%)')
        ax3.set_yticks(list(np.arange(0, 1.1, 0.1)))
        ax3.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        # ax3.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        ax3.legend(loc='lower right', fontsize='xx-small', frameon=True)
        ax3.spines['top'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        # ax3.spines['right'].set_visible(False)

        # Instantiate a second axes that shares the same x-axis
        ax3_twin = ax3.twinx()
        ax3_twin.set_ylim([0, 1.1])
        ax3_twin.set_yticks(list(np.arange(0, 1.1, 0.1)))
        ax3_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        ax3_twin.spines['top'].set_visible(False)
        ax3_twin.spines['bottom'].set_visible(False)

        time_end_miss = time.time()
        runtime_miss = time_end_miss - time_start_miss
        print("'Plot 3: misses' took", round(runtime_miss, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 4: HIT SCATTER PLOT

        time_start_hit = time.time()

        # Prepares the grid for the plots
        if df.Stage.unique()[0] == 4:
            ax4 = plt.subplot2grid((16, 4), (6, 0), rowspan=3, colspan=4)
        else:
            ax4 = plt.subplot2grid((16, 4), (12, 0), rowspan=4, colspan=4)
        # ax4 = plt.subplot2grid((4, 1), (3, 0))

        palette = ['tab:red', 'tab:green', 'grey']
        hue = ['Error' if i == 0 else 'Hit' if i == 1 else 'Miss' for i in df.Hit]
        hue_order = ['Error', 'Hit', 'Miss']

        if df.Stage.unique()[0] <= 3:  # No coherences, plot sides
            scatter = sns.scatterplot(x=df.index, y=df.Side, hue=hue, palette=palette,
                                      hue_order=hue_order,
                                      s=ms ** 2)
            ax4.set_ylim(-0.8, 1.8)
            ax4.set_yticks([0, 1])
            ax4.set_yticklabels(['L', 'R'])
            ax4.set_ylabel('Sides')

            # Instantiate a second axes that shares the same x-axis
            ax4_twin = ax4.twinx()
            ax4_twin.set_ylim(-0.8, 1.8)  # Evidences
            ax4_twin.set_yticks([0, 1])
            ax4_twin.set_yticklabels(['L', 'R'])
            ax4_twin.spines['top'].set_visible(False)

        else:  # Plot coherences
            scatter = sns.scatterplot(x=df.index, y=df.Evidence, hue=hue, palette=palette,
                                      hue_order=hue_order, s=ms ** 2)
            # Plot horizontal lines
            ax4.axhline(0, color='tab:gray', linestyle='--')  # Evidence 0
            ax4.axhline(-0.5, color='tab:gray', linestyle=':')  # Evidence -0.5
            ax4.axhline(0.5, color='tab:gray', linestyle=':')  # Evidence 0.5
            ax4.set_ylim(-1.1, 1.1)  # Evidences
            # ax.set_ylim(0, 1)  # Coherences
            ax4.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
            ax4.set_yticklabels(['L', '', '', '', '0', '', '', '', 'R'])  # Evidences
            # ax.set_yticklabels(['Left', '', '', '', '0.5', '', '',  '', 'Right'])  # Coherences
            ax4.set_ylabel('Evi.')
            # ax.set_ylabel('Coherence')

            # Instantiate a second axes that shares the same x-axis
            ax4_twin = ax4.twinx()
            ax4_twin.set_ylim(-1.1, 1.1)  # Evidences
            ax4_twin.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
            ax4_twin.set_yticklabels(['', '', '', '', '', '', '', '', ''])
            ax4_twin.spines['top'].set_visible(False)

        ax4.set_xlim([1, len(df)])  # 1 to not plot trial 0
        ax4.set_xlabel('Trial')
        # scatter.legend(bbox_to_anchor=(1, 1))
        scatter.legend(loc='lower right', fontsize='xx-small', frameon=True)
        # scatter.get_legend().remove()
        ax4.spines['top'].set_visible(False)
        # ax4.spines['right'].set_visible(False)

        time_end_hit = time.time()
        runtime_hit = time_end_hit - time_start_hit
        print("'Plot 4: misses' took", round(runtime_hit, 2), 'seconds to run')

        ################################################################################################################

        # PLOTS 8-9: PSYCOMETRIC CURVES

        # Only draw PC if evidences are introduced (stage 4)
        if len(df.Evidence.unique()) > 2 and df.Stage.unique()[0] == 4:
            # fig = plt.figure()

            # Psychometric curve of the whole session (all trials)
            ax11 = plt.subplot2grid((16, 4), (10, 0), rowspan=6, colspan=2)

            # Compute psychometric curves
            psych_curve = compute_psych_curve(df.Evidence, df.Choice)  # No need to filter out the misses
            psych_curve_rep = compute_psych_curve(df.EviRep, df.RepChoice)

            # Plot horizontal and vertical lines
            ax11.axhline(0.5, color='tab:gray', ls='--')
            ax11.axvline(0., color='tab:gray', ls='--')

            # Plot left-right psychometric curve and errorbars
            ax11.plot(np.linspace(-1, 1, 30), psych_curve.fit, color='tab:orange', label='L-R')
            ax11.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color='tab:orange', fmt='o',
                          markerfacecolor='none')

            # # Plot alt-rep psychometric curve and errorbars
            # ax11.plot(np.linspace(-1, 1, 30), psych_curve_rep.fit, color='tab:brown', label='Alt-Rep')
            # ax11.errorbar(psych_curve_rep.xdata, psych_curve_rep.ydata, yerr=psych_curve_rep.fit_error,
            #               color='tab:brown', fmt='o', markerfacecolor='none')

            ax11.set_xlabel('Evi.')
            ax11.set_xlim([-1.05, 1.05])
            ax11.set_ylabel('Prob. right')
            ax11.set_ylim([-0.025, 1.025])
            # ax11.set_yticks(np.arange(0, 1.1, step=0.1))
            ax11.legend(loc="lower right", frameon=False)

            # ax11_right_yaxis = ax11.twinx()  # instantiate a second axes that shares the same x-axis
            # # ax11_right_yaxis.set_ylabel('Prob. right')
            # ax11.set_yticklabels([])  # Remove left yticklabels
            # ax11.set_yticks([])  # Remove left yticks

            ax11.spines['top'].set_visible(False)
            ax11.spines['right'].set_visible(False)
            # ax11_right_yaxis.spines['top'].set_visible(False)
            # ax11_right_yaxis.spines['left'].set_visible(False)

            ax11.annotate(str(round(psych_curve.ydata[0], 2)), xy=(psych_curve.xdata[0], psych_curve.ydata[0]),
                          xytext=(psych_curve.xdata[0], psych_curve.ydata[0]), color='tab:red')
            ax11.annotate(str(round(psych_curve.ydata[-1], 2)), xy=(psych_curve.xdata[-1], psych_curve.ydata[-1]),
                          xytext=(psych_curve.xdata[-1], psych_curve.ydata[-1]), color='tab:red')

            sensitivity, bias, lr_left, lr_right = psych_curve.params

            ax11.annotate("S=" + str(round(sensitivity, 2)) + "\n" +  # Sensitivity
                          "B=" + str(round(bias, 2)) + "\n" +  # Bias
                          "LR_L=" + str(round(lr_left, 2)) + "\n" +  # Left lapse rate
                          "LR_R=" + str(round(lr_right, 2)), xy=(0, 0), xytext=(-1, 0.5),  # Right lapse rate
                          fontsize='xx-small')

            if df.Progression.unique()[0] == 1:

                # Psychometric curves per substage
                ax_list = [plt.subplot2grid((16, 6), (10, 3), rowspan=2, colspan=1),  # Substage 1
                           plt.subplot2grid((16, 6), (10, 4), rowspan=2, colspan=1),  # Substage 2
                           plt.subplot2grid((16, 6), (10, 5), rowspan=2, colspan=1),  # Substage 3
                           plt.subplot2grid((16, 6), (12, 3), rowspan=2, colspan=1),  # Substage 4
                           plt.subplot2grid((16, 6), (12, 4), rowspan=2, colspan=1),  # Substage 5
                           plt.subplot2grid((16, 6), (12, 5), rowspan=2, colspan=1),  # Substage 6
                           plt.subplot2grid((16, 6), (14, 3), rowspan=2, colspan=1),  # Substage 7
                           plt.subplot2grid((16, 6), (14, 4), rowspan=2, colspan=1),  # Substage 8
                           plt.subplot2grid((16, 6), (14, 5), rowspan=2, colspan=1)]  # Substage 9

                substages_session = df.Substage.unique()
                df_substages = {}  # Create empty dictionary

                for i in range(len(ax_list)):
                    if i + 1 not in substages_session:  # +1 because ax_list is from 0 to 8 and substages can go from 1 to 9
                        ax_list[i].set_visible(False)  # Make axes invisible
                        continue
                    else:
                        df_substages[i + 1] = df[df.Substage == i + 1]

                    # Compute psychometric curves
                    psych_curve = compute_psych_curve(
                        df_substages[i + 1].Evidence[df_substages[i + 1].Miss == 0],
                        df_substages[i + 1].Choice[df_substages[i + 1].Miss == 0])

                    ax = ax_list[i]

                    # Plot horizontal and vertical lines
                    ax.axhline(0.5, color='tab:gray', ls='--')
                    ax.axvline(0., color='tab:gray', ls='--')

                    # Plot left-right psychometric curve and errorbars
                    ax.plot(np.linspace(-1, 1, 30), psych_curve.fit, color='tab:orange', label='L-R')
                    ax.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color='tab:orange',
                                fmt='o', markerfacecolor='none')

                    ax.set_title(f'Sub.{i + 1}, n={len(df_substages[i + 1])}')
                    ax.set_xlim([-1.1, 1.1])  # Important so set_aspect can work for all subplots the same
                    ax.set_xticks([], [])
                    ax.set_ylim([-0.1, 1.1])
                    ax.set_yticks([], [])
                    # plt.axis('equal')

                    # x0, x1 = plt.gca().get_xlim()
                    # y0, y1 = plt.gca().get_ylim()
                    # plt.gca().set_aspect((x1 - x0) / (y1 - y0))  # Height is float times the width

                # plt.tight_layout()

                # This won't work unless updating matplotlib
                # plt.suptitle('Substage')
                # fig.supxlabel('Prob. right')
                # fig.supylabel('Evidence')

        ################################################################################################################

        time_start_savepag1 = time.time()
        pdf.savefig()  # saves the current figure into a pdf page
        time_end_savepag1 = time.time()
        runtime_savepag1 = time_end_savepag1 - time_start_savepag1
        print("'Saving 1st page in pdf' took", round(runtime_savepag1, 2), 'seconds to run')

        plt.close()

        ################################################################################################################
        ################################################################################################################

        # PAGE 2

        fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches portrait
        # fig = plt.figure(figsize=(11.69, 8.27))  # A4 size in inches landscape

        # PLOT 5: PERISTIMULUS LICK RASTER

        time_start_raster = time.time()

        # fig = plt.figure()
        xlim = [[], []]  # Initialize empty list to store left and right xlim

        for k in range(len(df.Side.unique())):  # k=0 left trials and k=1 right trials

            if k == 0:  # Left subplot: left trials
                # ax = plt.subplot2grid((1, 2), (0, 0))
                ax = plt.subplot2grid((16, 4), (0, 0), rowspan=12, colspan=2)
                stim_color = 'tab:blue'
                ax.set_title('Left trials')
                # ax.set_xlabel('Time (s)')
                ax.set_xticklabels([])
                ax.set_ylabel('Trial')
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:  # Right subplot: right trials
                # ax = plt.subplot2grid((1, 2), (0, 1))
                ax = plt.subplot2grid((16, 4), (0, 2), rowspan=12, colspan=2)
                stim_color = 'tab:orange'
                ax.set_title('Right trials')
                # ax.set_xlabel('Time (s)')
                ax.set_xticklabels([])
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['right'].set_visible(False)

            df_side = df[df.Side == k].reset_index()

            for j in range(len(df_side)):  # n trials

                # Plot stimulus length
                ax.barh(df_side.index.array[j],
                        df_side.StimLen[j],
                        left=df_side.StimStart[j] -
                             df_side.StimStart[j],
                        color=stim_color, alpha=1, label='Stim', zorder=1)  # Need to specify zorder otherwise
                # response window is plotted under stimulus length and can't be seen

                # Define response window color according to trial outcome
                if df_side.WrongLick[j] == 1.0:
                    resp_win_color = 'tab:pink'
                elif df_side.Hit[j] == 0.0:
                    resp_win_color = 'tab:red'
                elif df_side.Hit[j] == 1.0:
                    resp_win_color = 'tab:green'
                elif np.isnan(df_side.Hit[j]):
                    resp_win_color = 'tab:gray'

                # Plot response window length
                ax.barh(df_side.index.values[j],
                        df_side.RespWinLen[j],
                        left=df_side.RespWinStart[j] -
                             df_side.StimStart[j],
                        color=resp_win_color, zorder=2)

                # Left licks
                for i in range(len(df_side.Port1In[j])):  # n licks

                    # If licks are after stimulus onset, draw markeredgecolor so it can be seen over stimulus length barh
                    if df_side.Port1In[j][i] - \
                            df_side.StimStart[j] > \
                            df_side.StimStart[j] - \
                            df_side.StimStart[j]:
                        ms = 2
                        mec = 'k'
                        mew = 0.1
                    else:
                        ms = 1
                        mec = 'tab:blue'
                        mew = None

                    if df_side.Port1In[j] == []:
                        # if not df.Port1In[j]:  # Equivalent
                        pass
                    else:
                        ax.plot(df_side.Port1In[j][i] -
                                df_side.StimStart[j],
                                df_side.index[j], marker='o', ms=ms,
                                mec=mec, mew=mew, color='tab:blue', zorder=3)
                        # markersize=200 / len(df.Side == 0))
                        # markersize = ax.containers[1][0].get_height()

                # Right licks
                for i in range(len(df_side.Port2In[j])):  # n licks

                    # If licks are after stimulus onset, draw markeredgecolor so it can be seen over stimulus length barh
                    if df_side.Port2In[j][i] - \
                            df_side.StimStart[j] > \
                            df_side.StimStart[j] - \
                            df_side.StimStart[j]:
                        ms = 2
                        mec = 'k'
                        mew = 0.1
                    else:
                        ms = 1
                        mec = 'tab:orange'
                        mew = None

                    if df_side.Port2In[j] == []:
                        # if not df.Port1In[j]:  # Equivalent
                        pass
                    else:
                        ax.plot(df_side.Port2In[j][i] -
                                df_side.StimStart[j],
                                df_side.Port2In.index[j], marker='o', ms=ms,
                                mec=mec, mew=mew, color='tab:orange', zorder=3)
                        # markersize=200 / len(df.Side == 1))
                        # markersize = ax.containers[1][0].get_height()

            xlim[k] = [ax.get_xlim()]  # Store xlim from left and right plots
            ax.set_xlim([-2, xlim[k][0][1]])  # Set xlim from -2 to trial end to zoom in and cut the fixation

        # Custom legend
        legend_elements = [Patch(facecolor='tab:blue', label='Stim. left'),
                           Patch(facecolor='tab:orange', label='Stim. right'),
                           Patch(facecolor='tab:green', label='Correct'),
                           Patch(facecolor='tab:red', label='Error'),
                           Patch(facecolor='tab:pink', label='WrongLick'),
                           Patch(facecolor='tab:gray', label='Miss'),
                           Line2D([0], [0], marker='o', color='w', label='Left licks', markerfacecolor='tab:blue'),
                           Line2D([0], [0], marker='o', color='w', label='Right licks', markerfacecolor='tab:orange')]

        ax.legend(handles=legend_elements, loc='upper right', fontsize='xx-small', frameon=True)

        time_end_raster = time.time()
        runtime_raster = time_end_raster - time_start_raster
        print("'Plot 5: peristimulus lick raster' took", round(runtime_raster, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 6: PERISTIMULUS ALL LICKS HISTOGRAM

        time_start_psth_all = time.time()

        # fig = plt.figure()

        bin_size = 0.1

        for k in range(len(df.Side.unique())):  # k=0 left trials and k=1 right trials

            histcounts_L = []
            histcounts_R = []

            if k == 0:  # Left subplot: left trials
                # ax = plt.subplot2grid((1, 2), (0, 0))
                ax = plt.subplot2grid((16, 4), (12, 0), rowspan=2, colspan=2)
                # ax.set_title('Left trials')
                # ax.set_xlabel('Time (s)')
                ax.set_xlim(xlim[k][0])  # Use the same xlim that left raster
                # ax.set_xticklabels([])
                ax.set_ylabel('All licks\n(licks/s)')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:  # Right subplot: right trials
                # ax = plt.subplot2grid((1, 2), (0, 1))
                ax = plt.subplot2grid((16, 4), (12, 2), rowspan=2, colspan=2)
                # ax.set_title('Right trials')
                # ax.set_xlabel('Time (s)')
                ax.set_xlim(xlim[k][0])  # Use the same xlim that right raster
                # ax.set_xticklabels([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            df_side = df[df.Side == k].reset_index()

            for j in range(len(df_side)):  # n trials

                # Left licks
                for i in range(len(df_side.Port1In[j])):  # n licks
                    if df_side.Port1In[j] == []:
                        # if not df.Port1In[j]:  # Equivalent
                        pass
                    else:
                        histcounts_L.append(df_side.Port1In[j][i] -
                                            df_side.StimStart[j])

                # Right licks
                for i in range(len(df_side.Port2In[j])):  # n licks
                    if df_side.Port2In[j] == []:
                        # if not df.Port1In[j]:  # Equivalent
                        pass
                    else:
                        histcounts_R.append(df_side.Port2In[j][i] -
                                            df_side.StimStart[j])

            # ax.hist(histcounts_L, density=True, histtype='step', color='tab:blue', label='Left licks')
            # ax.hist(histcounts_R, density=True, histtype='step', color='tab:orange', label='Right licks')

            ax.hist(histcounts_L, histtype='step', color='tab:blue', label='Left licks',
                    bins=np.linspace(-2, xlim[k][0][1]),
                    weights=np.repeat((1 / len(df[(df.Miss == 0) & (df.Side == 0)])) / bin_size,
                                      len(histcounts_L)))
            ax.hist(histcounts_R, histtype='step', color='tab:orange', label='Right licks',
                    bins=np.linspace(-2, xlim[k][0][1]),
                    weights=np.repeat((1 / len(df[(df.Miss == 0) & (df.Side == 1)])) / bin_size,
                                      len(histcounts_R)))

            ax.set_xlim([-2, xlim[k][0][1]])  # Set xlim from -2 to trial end to zoom in and cut the fixation

        ax.legend(loc='upper right', fontsize='xx-small', frameon=True)

        time_end_psth_all = time.time()
        runtime_psth_all = time_end_psth_all - time_start_psth_all
        print("'Plot 6: peristimulus lick histogram (all licks)' took", round(runtime_psth_all, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 7: PERISTIMULUS FIRST LICK HISTOGRAM

        time_start_psth_first = time.time()

        # fig = plt.figure()

        for k in range(len(df.Side.unique())):  # k=0 left trials and k=1 right trials

            first_lick_L = []
            first_lick_R = []

            if k == 0:  # Left subplot: left trials
                # ax = plt.subplot2grid((1, 2), (0, 0))
                ax = plt.subplot2grid((16, 4), (15, 0), rowspan=1, colspan=2)
                # ax.set_title('Left trials')
                ax.set_xlim([0, 2])  # Only interested in what happens during the first 2s
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('First lick\n(licks/s)')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # ax.patch.set_facecolor('none')
            else:  # Right subplot: right trials
                # ax = plt.subplot2grid((1, 2), (0, 1))
                ax = plt.subplot2grid((16, 4), (15, 2), rowspan=1, colspan=2)
                # ax.set_title('Right trials')
                ax.set_xlim([0, 2])
                ax.set_xlabel('Time (s)')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # ax.patch.set_facecolor('none')

            df_side = df[df.Side == k].reset_index()

            for j in range(len(df_side)):  # n trials

                # Left licks
                for i in range(len(df_side.Port1In[j])):  # n licks
                    if df_side.Port1In[j] == []:
                        # if not df.Port1In[j]:  # Equivalent
                        pass
                    else:
                        if df_side.Port1In[j][i] - \
                                df_side.StimStart[j] < df.RespWinStart[
                            j] - df_side.StimStart[j]:
                            first_lick_L.append(df_side.Port1In[j][i] -
                                                df_side.StimStart[j])
                        elif df_side.Port1In[j][i] - \
                                df_side.StimStart[j] > df.RespWinStart[
                            j] - df_side.StimStart[j]:
                            first_lick_L.append(df_side.Port1In[j][i] -
                                                df_side.StimStart[j])
                            break

                # Right licks
                for i in range(len(df_side.Port2In[j])):  # n licks
                    if df_side.Port2In[j] == []:
                        # if not df.Port1In[j]:  # Equivalent
                        pass
                    else:
                        if df_side.Port2In[j][i] - \
                                df_side.StimStart[j] < df.RespWinStart[
                            j] - df_side.StimStart[j]:
                            first_lick_R.append(df_side.Port2In[j][i] -
                                                df_side.StimStart[j])
                        elif df_side.Port2In[j][i] - \
                                df_side.StimStart[j] > df.RespWinStart[
                            j] - df_side.StimStart[j]:
                            first_lick_R.append(df_side.Port2In[j][i] -
                                                df_side.StimStart[j])
                            break

            # ax.hist(first_lick_L, density=True, histtype='step', color='tab:blue', label='Left')
            # ax.hist(first_lick_R, density=True, histtype='step', color='tab:orange', label='Right')

            ax.hist(first_lick_L, histtype='step', color='tab:blue', label='Left licks', bins=np.linspace(0, 2),
                    weights=np.repeat((1 / len(df[(df.Miss == 0) & (df.Side == 0)])) / bin_size,
                                      len(first_lick_L)))
            ax.hist(first_lick_R, histtype='step', color='tab:orange', label='Right licks',
                    bins=np.linspace(0, 2),
                    weights=np.repeat((1 / len(df[(df.Miss == 0) & (df.Side == 1)])) / bin_size,
                                      len(first_lick_R)))

            ax.patch.set_facecolor('none')  # Make axes transparent so the xaxes labels from the upper plot are visible
        # ax.legend(loc='upper right')

        time_end_psth_first = time.time()
        runtime_psth_first = time_end_psth_first - time_start_psth_first
        print("'Plot 7: peristimulus lick histogram (first licks)' took", round(runtime_psth_first, 2),
              'seconds to run')

        time_start_savepag2 = time.time()
        pdf.savefig()  # saves the current figure into a pdf page
        time_end_savepag2 = time.time()
        runtime_savepag2 = time_end_savepag2 - time_start_savepag2
        print("'Saving 2nd page in pdf' took", round(runtime_savepag2, 2), 'seconds to run')

        plt.close()

    ####################################################################################################################

    # Register time again and compute the total run time of the script
    time_end_total = time.time()
    runtime_total = time_end_total - time_start_total
    print('The script took', round(runtime_total, 2), 'seconds to run', '\n')

    # This block needs to be the last otherwise it sends the file too soon and corrupted
    if send_slack:
        with open('/home/alexis/slack_bot_token', 'r') as f:  # Get slack bot token
            slack_bot_token = f.read().replace('\n', '')

        os.environ['SLACK_BOT_TOKEN'] = slack_bot_token
        filepath = folder + '/' + df.Session.unique()[0]
        slack_spam(msg='Hey buddy!', filepath=filepath, userid='#pv_nmdar_eranet')  # Alexis: 'U01DDHH7LLX'


########################################################################################################################

# To do:
# Select what sessions to do the reports
# Choose between plotting together sessions from the same day (with maybe a vertical red line) to separate them, or
# in separate reports

# Line of bash code to sync the cluster data with the local machine:
# rsync -avzP -e 'ssh -p 4022' mouse@neurocomp.fcrb.es:/archive/mouse/pv_nmdar_eranet* ~/
# rsync -avzP -e 'ssh -p 4022' mouse@neurocomp.fcrb.es:/archive/mouse/pv_nmdar_eranet* ~/ && rsync -avzP -e 'ssh -p 4022' mouse@neurocomp.fcrb.es:/archive/mouse/pluginsr-for-pybpod* ~/ && rsync -avzP -e 'ssh -p 4022' mouse@neurocomp.fcrb.es:/archive/mouse/pybpod_changes* ~/


# if __name__ == "__main__":
#     daily_report()
