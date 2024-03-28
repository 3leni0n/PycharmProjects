# To do:
# Check using GridSpec instead of plt.subplot2grid as suggested by matplotlib doc
# (https://matplotlib.org/stable/gallery/userdemo/demo_gridspec01.html)
# Make code detect OS and fill the destiny path automatically
# Select what sessions to do the reports (GUI)
# Choose between plotting together sessions from the same day (with maybe a vertical red line) to separate them, or
# in separate reports

# If no blocks, accuracy blocks = None/nan

########################################################################################################################

import time
from pathlib import Path
import os
import matplotlib.pyplot as plt
# plt.switch_backend('agg')  # To prevent RuntimeError: Invalid DISPLAY variable when running the script within a bash
# through crontab
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # For inset plot
from matplotlib.patches import Patch  # For custom legend
from matplotlib.lines import Line2D  # For custom legend
from matplotlib.backends.backend_pdf import PdfPages  # For saving figure as pdf
import seaborn as sns
import numpy as np
import pandas as pd

from my_fun.my_fun import compute_window, compute_psych_curve, slack_spam  # Or from my_fun.my_fun import my_fun
from parse.parse_v2 import *


########################################################################################################################
# To do list
# Include new VARs in the upper text: ITI, Recovery, Blocks, Block length, Block accuracy, Resp win, stim_sur, delay
# Collapse all sound errors into one to make space. Done

########################################################################################################################

# Define function
def daily_report_v5(path, send_slack=False):

    # Register time
    time_start_total = time.time()

    ####################################################################################################################

    # Import session to be parsed
    df = parse_v2(path)

    ####################################################################################################################

    # Import session to be parsed
    sounds = pd.read_csv(Path.home() / 'PycharmProjects' / 'create_sounds' / 'sounds_2.csv')
    ilds = list(sounds.ILD.unique())

    ####################################################################################################################

    # Select the folder and create it if it doesn't exist
    experiment = df.Experiment.unique()[0]  # Batch ID
    folder = Path.home() / 'Documents' / 'daily reports' / experiment
    if not os.path.exists(folder):
        # os.mkdir(folder)
        folder.mkdir(parents=True, exist_ok=True)
    os.chdir(folder)
    setup = df.Setup.unique()[0]  # Animal ID
    # folder = folder + setup
    folder = Path(folder / setup)
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

    # Responses (valid trials)
    responses = df.Response.sum()
    responses_left = df.Response[df.Side == 0].sum()
    responses_right = df.Response[df.Side == 1].sum()

    # Performance (response rate)
    performance = responses / trials
    performance_left = responses_left / trials_left
    performance_right = responses_right / trials_right

    # Accuracy (hit rate)
    accuracy = hits / responses
    accuracy_left = hits_left / responses_left
    accuracy_right = hits_right / responses_right

    # Block accuracy (accuracy of first trial of each block)
    if not pd.isnull(df.Blocks.unique()[0]) or int(df.Blocks.unique()[0]) != 0:  # If blocks isn't NaN or 0
        block_change_index = [i for i in range(1, len(df.Side)) if df.Side[i - 1] != df.Side[i]]
        # if not pd.isnull(df.BlockLen.unique()[0]) and float(df.BlockLen.unique()[0]) != 0:
            # BlockLen wasn't there from the beginning of blocks (block length = running window in the task = 20 trials),
            # so the previous method is a way to detect blocks regardless of the BlockLen. Nevertheless, in sessions
            # where there was block length, it should match with the previous method
            # Assertion not valid if transitioning from blocks to random trials. Need to include Warming up blocks
            # assert block_change_index == df.Side[
            #                              int(df.BlockLen.unique()[0])::int(
            #                                  df.BlockLen.unique()[0])].index.values.tolist()
            # pass

        hits_blocks = df.Hit[block_change_index].sum().astype(int)
        responses_blocks = df.Response[block_change_index].sum()
        accuracy_blocks = hits_blocks / responses_blocks
        hits_blocks_left = df.Hit[block_change_index][df.Side == 0].sum().astype(int)
        responses_block_left = df.Response[block_change_index][df.Side == 0].sum()
        accuracy_blocks_left = hits_blocks_left / responses_block_left
        hits_blocks_right = df.Hit[block_change_index][df.Side == 1].sum().astype(int)
        responses_block_right = df.Response[block_change_index][df.Side == 1].sum()
        accuracy_blocks_right = hits_blocks_right / responses_block_right
    else:
        hits_blocks = responses_blocks = accuracy_blocks = hits_blocks_left = responses_blocks_left = \
            accuracy_blocks_left = hits_blocks_right = responses_blocks_right = accuracy_blocks_right = np.nan

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
    sound_errors = sounds_mismatch + no_sound + message_count

    ####################################################################################################################

    with PdfPages(df.Session.unique()[0]) as pdf:

        # PAGE 1

        fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches portrait
        # fig = plt.figure(figsize=(11.69, 8.27))  # A4 size in inches landscape
        # fig = plt.figure()

        ################################################################################################################

        # SUMMARY TEXT
        new_line = '\n'  # Trick to include new lines in formatted strings
        # https://towardsdatascience.com/how-to-add-new-line-in-python-f-strings-7b4ccc605f4a
        sum_text = (
            f'Date: {df.Date.unique()[0]}, '
             # [0:-7] to get rid of the floating numbers in the seconds
            f'Time: {df.SessionStart.unique()[0][0:-7]} - {df.SessionEnd.unique()[0][0:-7]}, '
            f'Subject: {df.Subject.unique()[0]}, '
            f'Box: {df.Board.unique()[0][4]}, '
            # f'Stage: {str(df.Stage.unique()[0])}, '  # Legacy
            # f'Fixation: {str(df.Fixation.unique()[0])},'  # Legacy
            f'Switch: {str(df.Switch.unique()[0])}, '
            f'Timeout: {str(df.Timeout.unique()[0])}, '
            f'{new_line}'
            # f'Motor: {str(df.Motor.unique()[0])}, '  # Legacy
            # f'REC: {str(df.REC.unique()[0])}, '  # Always 1, useless
            f'CB: {str(df.CB.unique()[0])}, '
            f'Warm up: {str(df.WarmUp.unique()[0])}, '
            f'Progression: {str(df.Progression.unique()[0])}, '
            f'P: {str(round(df.P.iloc[1], 2))}, '
            f'P right: {str(round(df.PRight.iloc[1], 2))}, '
            f'ITI: {df.ITI.unique()[0]}, '
            f'Recovery: {df.RecoveryMode.unique()[0]}, '
            f'Blocks: {df.Blocks.unique()[0]}, '
            f'Block '
            f'{new_line}'
            f'length: {int(df.BlockLen.unique()[0])}, '
            f'Response window: {df.RespWin.unique()[0]}s, '
            f'Stimulus duration: {df.StimDur.unique()[0]}s, '
            f'Delay: {round(df.VarDelay.mean(), 1)}s, '
            # f'Delay: {df.Delay.unique()[0]}s, '
            f'Task: {df.Task.unique()[0]}, '
            f'Trials: {str(trials)} '
            f'{new_line}'
            f'({str(trials_left)} L, {str(trials_right)} R), '
            f'Performance: {str(int(round(performance * 100)))}% ({str(int(round(performance_left * 100)))}% L, '
            f'{str(int(round(performance_right * 100)))}% R), '
            f'Accuracy: {str(int(round(accuracy * 100)))}% ({str(int(round(accuracy_left * 100)))}% L, '
            f'{str(int(round(accuracy_right * 100)))}% R), '
            f'Accuracy '
            f'{new_line}'
            f'blocks: {str(int(round(accuracy_blocks * 100)))}% ({str(int(round(accuracy_blocks_left * 100)))}% L, '
            f'{str(int(round(accuracy_blocks_right * 100)))}% R),'
            f'Responses: {str(responses)} ({str(responses_left)} L, {str(responses_right)} R), '
            f'Hits: {str(hits)} ({str(hits_left)} L, {str(hits_right)} R), '
            f'Errors: '
            f'{new_line}'
            f'{str(errors)} ({str(errors_left)} L, {str(errors_right)} R), '
            f'Misses: {str(misses)} ({str(misses_left)} L, {str(misses_right)} R), '
            f'Miss rate: {str(int(round(miss_rate * 100)))}% ' 
            f'({str(int(round(miss_rate_left * 100)))}% L, {str(int(round(miss_rate_right * 100)))}% R), '
            f'Sound errors: {str(sound_errors)} ({str(round((sound_errors / trials) * 100, 1))}%), '
            f'{new_line}'
            f'AW: {str(df.AW.unique()[0])} trials, '
            f'Water: {str(water)} μL ({str(water_left)}μL L {str(water_right)}μL R), '
            f'Wait: {df.Wait.unique()[0]} min.'
            f'{new_line}'
            f'{new_line}')

        # plt.text(0.1, 0.90, sum_text, fontsize=8, transform=plt.gcf().transFigure)

        ################################################################################################################

        # fig = plt.figure()

        change_p = df.P.diff()  # Find trials in which substage/p changes
        change_p = change_p[change_p != 0].dropna()  # Omit 0s and drop first nan

        # PLOT 1: ACCURACY PER SIDE

        time_start_acc_side = time.time()

        # Compute accuracy rolling average
        ra_total = compute_window(df.Hit[df.Miss == 0], 20)  # All valid trials
        ra_left = compute_window(df.Hit[(df.Miss == 0) & (df.Side == 0)],
                                 20)  # Left valid trials
        ra_right = compute_window(df.Hit[(df.Miss == 0) & (df.Side == 1)],
                                  20)  # Right valid trials

        # # Prepares the grid for the psychometric_curves
        # ax1 = plt.subplot2grid((16, 4), (0, 0), rowspan=4, colspan=4)
        # # ax1 = plt.subplot2grid((4, 1), (0, 0))

        # Prepares the grid for the psychometric_curves
        # if df.Stage.unique()[0] == 4:  # Legacy
        if df.Task.unique()[0] == 'FD' and df.P.unique()[0] > 0:
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
            for i in range(len(change_p.index)):
                if change_p[change_p.index[i]] > 0:
                    # ax1.annotate(s='', xy=(change_p.index[i], 1), xytext=(change_p.index[i], 0),
                    #              arrowprops=dict(arrowstyle='->', color='green'))
                    ax1.plot(change_p.index[i], 0.1, marker='^', ms=ms, lw=lw, color='tab:green')
                    ax1.annotate(str(round(df.P[change_p.index[i]], 2))[1:],
                                 xy=(change_p.index[i], 0.1), xytext=(change_p.index[i], 0.2),
                                 color='tab:green', ha='center')
                elif change_p[change_p.index[i]] < 0:
                    # ax1.annotate(s='', xy=(change_p.index[i], 1), xytext=(change_p.index[i], 0),
                    #              arrowprops=dict(arrowstyle='<-', color='red'))
                    ax1.plot(change_p.index[i], 0.1, marker='v', ms=ms, lw=lw, color='tab:red')
                    ax1.annotate(str(round(df.P[change_p.index[i]], 2))[1:],
                                 xy=(change_p.index[i], 0.1), xytext=(change_p.index[i], 0.2),
                                 color='tab:red', ha='center')

        ax1.set_xlim([1, len(df)])  # 1 to not plot trial 0
        ax1.set_xticklabels([])
        ax1.set_ylabel('Acc.\n(%)')
        ax1.set_ylim([0, 1.1])
        ax1.set_yticks(list(np.arange(0, 1.1, 0.1)))
        ax1.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        ax1.legend(loc='lower right', fontsize='xx-small', frameon=False)
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
        # ax1.text(0, 1, s1 + s2 + s3 + s4 + s5 + s6, fontsize='medium')
        ax1.text(0, 1, sum_text, fontsize='medium')

        ################################################################################################################

        # PLOT 2: REPEATING VS ALTERNATING ACCURACY

        time_start_acc_repalt = time.time()

        # Compute accuracy rolling average for repeating vs alternating trials
        ra_rep = compute_window(df.Hit[(df.Miss == 0) & (df.RepTrial == 1)], 20)
        ra_alt = compute_window(df.Hit[(df.Miss == 0) & (df.RepTrial == 0)], 20)

        # Prepares the grid for the psychometric_curves
        # if df.Stage.unique()[0] == 4:  # Legacy
        if df.Task.unique()[0] == 'FD' and df.P.unique()[0] > 0:
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
            for i in range(len(change_p.index)):
                if change_p[change_p.index[i]] > 0:
                    # ax2.annotate(s='', xy=(change_p.index[i], 1), xytext=(change_p.index[i], 0),
                    #              arrowprops=dict(arrowstyle='->', color='green'))
                    ax2.plot(change_p.index[i], 0.1, marker='^', ms=ms, lw=lw, color='tab:green')
                    ax2.annotate(str(round(df.P[change_p.index[i]], 2))[1:],
                                 xy=(change_p.index[i], 0.1), xytext=(change_p.index[i], 0.2),
                                 color='tab:green', ha='center')
                elif change_p[change_p.index[i]] < 0:
                    # ax2.annotate(s='', xy=(change_p.index[i], 1), xytext=(change_p.index[i], 0),
                    #              arrowprops=dict(arrowstyle='<-', color='red'))
                    ax2.plot(change_p.index[i], 0.1, marker='v', ms=ms, lw=lw, color='tab:red')
                    ax2.annotate(str(round(df.P[change_p.index[i]], 2))[1:],
                                 xy=(change_p.index[i], 0.1), xytext=(change_p.index[i], 0.2),
                                 color='tab:red', ha='center')

        ax2.set_xlim([1, len(df)])  # 1 to not plot trial 0
        ax2.set_xticklabels([])
        ax2.set_ylabel('Acc.\n(%)')
        ax2.set_ylim([0, 1.1])
        ax2.set_yticks(list(np.arange(0, 1.1, 0.1)))
        ax2.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        ax2.legend(loc='lower right', fontsize='xx-small', frameon=False)
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

        # Prepares the grid for the psychometric_curves
        # if df.Stage.unique()[0] == 4:  # Legacy
        if df.Task.unique()[0] == 'FD' and df.P.unique()[0] > 0:
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
            for i in range(len(change_p.index)):
                if change_p[change_p.index[i]] > 0:
                    # ax3.annotate(s='', xy=(change_p.index[i], 1), xytext=(change_p.index[i], 0),
                    #              arrowprops=dict(arrowstyle='->', color='green'))
                    ax3.plot(change_p.index[i], 0.1, marker='^', ms=ms, lw=lw, color='tab:green')
                    ax3.annotate(str(round(df.P[change_p.index[i]], 2))[1:],
                                 xy=(change_p.index[i], 0.1), xytext=(change_p.index[i], 0.2),
                                 color='tab:green', ha='center')
                elif change_p[change_p.index[i]] < 0:
                    # ax3.annotate(s='', xy=(change_p.index[i], 1), xytext=(change_p.index[i], 0),
                    #              arrowprops=dict(arrowstyle='<-', color='red'))
                    ax3.plot(change_p.index[i], 0.1, marker='v', ms=ms, lw=lw, color='tab:red')
                    ax3.annotate(str(round(df.P[change_p.index[i]], 2))[1:],
                                 xy=(change_p.index[i], 0.1), xytext=(change_p.index[i], 0.2),
                                 color='tab:red', ha='center')

        ax3.set_xlim([1, len(df)])  # 1 to not plot trial 0
        ax3.set_xticklabels([])
        ax3.set_ylim([0, 1.1])
        ax3.set_ylabel('Miss\n(%)')
        ax3.set_yticks(list(np.arange(0, 1.1, 0.1)))
        ax3.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        ax3.legend(loc='upper right', fontsize='xx-small', frameon=False)
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

        # Prepares the grid for the psychometric_curves
        # if df.Stage.unique()[0] == 4:  # Legacy
        if df.Task.unique()[0] == 'FD' and df.P.unique()[0] > 0:
            ax4 = plt.subplot2grid((16, 4), (6, 0), rowspan=3, colspan=4)
        else:
            ax4 = plt.subplot2grid((16, 4), (12, 0), rowspan=4, colspan=4)
        # ax4 = plt.subplot2grid((4, 1), (3, 0))

        # Plot horizontal lines
        ax4.axhline(0, color='tab:gray', linestyle='--')  # Evidence 0
        # ax4.axhline(-0.5, color='tab:gray', linestyle=':')  # Evidence -0.5
        # ax4.axhline(0.5, color='tab:gray', linestyle=':')  # Evidence 0.5

        palette = ['tab:red', 'tab:green', 'grey']
        hue = ['Error' if i == 0 else 'Hit' if i == 1 else 'Miss' for i in df.Hit]
        hue_order = ['Error', 'Hit', 'Miss']

        # if df.Stage.unique()[0] <= 3:  # Legacy
        if df.Task.unique()[0] == 'RT':  # No coherences, plot sides
            scatter = sns.scatterplot(x=df.index, y=df.Side, hue=hue, palette=palette,
                                      hue_order=hue_order, s=ms ** 3,
                                      zorder=2.5)  # zorder=2.5 to plot the dots over the line
            ax4.set_ylim(-0.8, 1.8)
            ax4.set_yticks([0, 1])
            ax4.set_yticklabels(['L', 'R'])
            ax4.set_ylabel('Side')

            # Instantiate a second axes that shares the same x-axis
            ax4_twin = ax4.twinx()
            ax4_twin.set_ylim(-0.8, 1.8)  # Evidences
            ax4_twin.set_yticks([0, 1])
            ax4_twin.set_yticklabels(['', ''])
            ax4_twin.spines['top'].set_visible(False)

        else:  # Plot coherences
            scatter = sns.scatterplot(x=df.index, y=df.ILD, hue=hue, palette=palette,
                                      hue_order=hue_order, s=ms ** 3,
                                      zorder=2.5)  # zorder=2.5 to plot the dots over the line

            ax4.set_yscale('symlog', linthresh=20)  # Set symmetric logarithmic spacing to zoom in the middle
            ax4.set_ylim(-100, 100)
            ax4.minorticks_off()  # Remove minor ticks
            # ilds = np.sort(df.ILD.unique().astype('int'))
            yticklabels = list(ilds)
            ax4.set_yticks(ilds)
            ax4.set_yticklabels(yticklabels)
            ax4.set_ylabel('ILD')

            # Instantiate a second axes that shares the same x-axis
            ax4_twin = ax4.twinx()
            ax4_twin.set_yscale('symlog', linthresh=20)  # Set symmetric logarithmic spacing to zoom in the middle
            ax4_twin.minorticks_off()  # Remove minor ticks
            ax4_twin.set_ylim(ax4.get_ylim())
            ax4_twin.set_yticks(ilds)
            ax4_twin.set_yticklabels(['', '', '', '', '', '', '', '', ''])

            ax4_twin.spines['top'].set_visible(False)

        ax4.set_xlim([1, len(df)])  # 1 to not plot trial 0
        ax4.set_xlabel('Trial')
        # scatter.legend(bbox_to_anchor=(1, 1))
        scatter.legend(loc='lower right', fontsize='xx-small', frameon=False)
        # scatter.get_legend().remove()
        ax4.spines['top'].set_visible(False)
        # ax4.spines['right'].set_visible(False)

        time_end_hit = time.time()
        runtime_hit = time_end_hit - time_start_hit
        print("'Plot 4: misses' took", round(runtime_hit, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 8: P RIGHT PSYCHOMETRIC CURVE

        # To do:
        # Change the dense x,y variables notation for annotate by just selecting beforehand which are the x,y coordinates

        # Only draw PC if evidences are introduced (stage 4)
        if len(df.ILD.unique()) > 2 and df.Task.unique()[0] == 'FD' and df.P.unique()[0] > 0:
            # fig = plt.figure()

            # Psychometric curve of the whole session (all trials)
            ax11 = plt.subplot2grid((16, 4), (10, 0), rowspan=6, colspan=2)

            # Compute prob. right psychometric curve
            psych_curve = compute_psych_curve(df.ILD, df.Choice)  # No need to filter out the misses

            # Plot horizontal and vertical lines
            ax11.axhline(0.5, color='tab:gray', ls='--')
            ax11.axvline(0., color='tab:gray', ls='--')

            # Plot left-right psychometric curve and errorbars
            ax11.plot(np.linspace(np.min(df.ILD), np.max(df.ILD), len(psych_curve.fit)), psych_curve.fit,
                      color='tab:orange', label='L-R')

            # Move extreme datapoints closer to the center to zoom in
            psych_curve.xdata[0] = -20
            psych_curve.xdata[-1] = 20

            ax11.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color='tab:orange', fmt='o',
                          markerfacecolor='none')

            # ax11.set_xscale('symlog', linthreshx=20)  # Set symmetric logarithmic spacing to zoom in the middle

            plt.xlim([-21, 21])  # To chop the extreme values
            ilds[0] = -20
            ilds[-1] = 20
            plt.xticks(ilds)
            plt.gca().set_xticklabels(['-70', '-8', '', '', '0', '', '', '8', '70'])
            plt.ylim([-0.025, 1.025])
            plt.yticks([0, 0.5, 1])

            ax11.minorticks_off()  # Remove minor ticks
            ax11.set_xlabel('ILD')
            ax11.set_ylabel('Prob. right')
            ax11.set_ylim([-0.025, 1.025])

            ax11.spines['top'].set_visible(False)
            ax11.spines['right'].set_visible(False)

            # Annotate min and max
            ax11.annotate(str(round(psych_curve.ydata[0], 2)), xy=(psych_curve.xdata[0], psych_curve.ydata[0]),
                          xytext=(psych_curve.xdata[0], psych_curve.ydata[0]), color='tab:orange', va='bottom',
                          ha='left', fontsize='medium')
            ax11.annotate(str(round(psych_curve.ydata[-1], 2)), xy=(psych_curve.xdata[-1], psych_curve.ydata[-1]),
                          xytext=(psych_curve.xdata[-1], psych_curve.ydata[-1]), color='tab:orange', va='top',
                          ha='right', fontsize='medium')

            sensitivity, bias, lr_left, lr_right = psych_curve.params  # Extract psychometric curve parameters

            # Annotate parameters
            ax11.annotate("S=" + str(round(sensitivity, 2)) + "\n" +  # Sensitivity
                          "B=" + str(round(bias, 2)) + "\n" +  # Bias
                          "LR_L=" + str(round(lr_left, 2)) + "\n" +  # Left lapse rate
                          "LR_R=" + str(round(lr_right, 2)),
                          xy=(-20, 1), xytext=(-20, 1), color='tab:orange', va='top', ha='left', fontsize='medium')

        ################################################################################################################

        # PLOT 9: P REPEAT PSYCHOMETRIC CURVE

        # To do:
        # Change the dense x,y variables notation for annotate by just selecting beforehand which are the x,y coordinates

        # Only draw PC if evidences are introduced (stage 4)
        if len(df.ILD.unique()) > 2 and df.Task.unique()[0] == 'FD' and df.P.unique()[0] > 0:
            # fig = plt.figure()

            # Psychometric curve of the whole session (all trials)
            ax13 = plt.subplot2grid((16, 4), (10, 2), rowspan=6, colspan=2)

            # Compute prob. rep psychometric curve
            psych_curve_rep = compute_psych_curve(df.ILDRep, df.RepChoice)

            # Plot horizontal and vertical lines
            ax13.axhline(0.5, color='tab:gray', ls='--')
            ax13.axvline(0., color='tab:gray', ls='--')

            # Plot alt-rep psychometric curve and errorbars
            ax13.plot(np.linspace(np.min(df.ILD), np.max(df.ILD), len(psych_curve.fit)), psych_curve_rep.fit,
                      color='tab:brown', label='Alt-Rep')

            # Move extreme datapoints closer to the center to zoom in
            psych_curve_rep.xdata[0] = -20
            psych_curve_rep.xdata[-1] = 20

            ax13.errorbar(psych_curve_rep.xdata, psych_curve_rep.ydata, yerr=psych_curve_rep.fit_error,
                          color='tab:brown', fmt='o', markerfacecolor='none')

            # ax13.set_xscale('symlog', linthreshx=20)  # Set symmetric logarithmic spacing to zoom in the middle

            plt.xlim([-21, 21])  # To chop the extreme values
            ilds[0] = -20
            ilds[-1] = 20
            plt.xticks(ilds)
            plt.gca().set_xticklabels(['-70', '-8', '', '', '0', '', '', '8', '70'])
            plt.ylim([-0.025, 1.025])
            plt.yticks([0, 0.5, 1])

            ax13.minorticks_off()  # Remove minor ticks
            ax13.set_xlabel('ILD')
            ax13.set_ylim([-0.025, 1.025])

            ax13_right_yaxis = ax13.twinx()  # instantiate a second axes that shares the same x-axis
            ax13_right_yaxis.set_ylabel('Prob. repeat')
            ax13.set_yticklabels([])  # Remove left yticklabels
            ax13.set_yticks([])  # Remove left yticks

            ax13.spines['top'].set_visible(False)
            ax13.spines['left'].set_visible(False)
            ax13_right_yaxis.spines['top'].set_visible(False)
            ax13_right_yaxis.spines['left'].set_visible(False)

            # Annotate min and max
            ax13.annotate(str(round(psych_curve_rep.ydata[0], 2)),
                          xy=(psych_curve_rep.xdata[0], psych_curve_rep.ydata[0]),
                          xytext=(psych_curve_rep.xdata[0], psych_curve_rep.ydata[0]), color='tab:brown', va='bottom',
                          ha='left', fontsize='medium')
            ax13.annotate(str(round(psych_curve_rep.ydata[-1], 2)),
                          xy=(psych_curve_rep.xdata[-1], psych_curve_rep.ydata[-1]),
                          xytext=(psych_curve_rep.xdata[-1], psych_curve_rep.ydata[-1]), color='tab:brown', va='top',
                          ha='right', fontsize='medium')

            sensitivity, bias, lr_left, lr_right = psych_curve_rep.params  # Extract psychometric curve parameters

            # Annotate parameters
            ax13.annotate("S=" + str(round(sensitivity, 2)) + "\n" +  # Sensitivity
                          "B=" + str(round(bias, 2)) + "\n" +  # Bias
                          "LR_Rep=" + str(round(lr_left, 2)) + "\n" +  # Left lapse rate
                          "LR_Alt=" + str(round(lr_right, 2)),
                          xy=(-20, 1), xytext=(-20, 1), color='tab:brown', va='top', ha='left', fontsize='medium')

        ################################################################################################################

        # PLOT 8: ILDS DISTRIBUTION

        # Only draw ILDs distribution if evidences are introduced (stage 4)
        if len(df.ILD.unique()) > 2 and df.Task.unique()[0] == 'FD' and df.P.unique()[0] > 0:
            # fig = plt.figure()

            # ILDs distribution of the whole session (all trials)
            # ax12 = plt.subplot2grid((16, 4), (10, 2), rowspan=6, colspan=2)
            # axins2 = inset_axes(ax13, width="50%", y2="50%", loc=4)
            ax12 = inset_axes(ax13, width="25%", height="25%", loc=4, borderpad=2)

            ax12.bar(0, len(df[df.ILD == -70]), color='k')
            ax12.bar(1, len(df[df.ILD == -8]), color='k')
            ax12.bar(2, len(df[df.ILD == -4]), color='k')
            ax12.bar(3, len(df[df.ILD == -2]), color='k')
            ax12.bar(4, len(df[df.ILD == -0]), color='k')
            ax12.bar(5, len(df[df.ILD == 2]), color='k')
            ax12.bar(6, len(df[df.ILD == 4]), color='k')
            ax12.bar(7, len(df[df.ILD == 8]), color='k')
            ax12.bar(8, len(df[df.ILD == 70]), color='k')
            ax12.set_xticks(np.arange(0, len(ilds)))
            # ax12.set_xticklabels(xticklabels)
            ax12.set_xticklabels(['-70', '', '', '', '0', '', '', '', '70'])
            ax12.patch.set_facecolor('none')  # Make background transparent
            # ax12.patch.set_alpha(0.0)  # Alternative
            plt.draw()  # Redraw the current figure so the background actually changes to transparent
            # ax12.set_xticks([])
            # ax12.set_xticklabels([])
            # ax12.set_yticks([])
            # ax12.set_yticklabels([])

            # ax12.spines['left'].set_visible(False)
            ax12.spines['right'].set_visible(False)
            # ax12.spines['bottom'].set_visible(False)
            ax12.spines['top'].set_visible(False)
            # ax12.axis('off')
            # ax12.set_frame_on(False)

            ax12.text(4, ax12.get_ylim()[1], '$\it{p_{mean}}$=' + str(round(df.P.mean(), 2)), color='k', va='top',
                      ha='center', fontsize='x-small')

            # # ax12.hist(df.ILD, bins=100, color='k')
            # ax12.hist(df.ILD, bins=range(len(ilds)), color='k')
            # ax12.set_xlim(100, -100)
            # ax12.set_xticks(np.sort(df.ILD.unique()))
            # ax12.set_yticklabels([])
            # ax12.spines['top'].set_visible(False)

            # # Instantiate a second axes that shares the same x-axis
            # ax12_twin = ax12.twinx()
            # ax12_twin.set_yticks(ax12.get_yticks())
            # ax12_twin.spines['top'].set_visible(False)
            # ax12_twin.spines['bottom'].set_visible(False)

            # ax12.text(0, ax12.get_ylim()[1], '$\it{p_{mean}}$=' + str(round(df.P.mean(), 2)), color='k', va='top',
            #           ha='center', fontsize='medium')

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

        # Set max xlim for lick rasters and histograms
        # xlim_max = df.RespWin.unique()[0] + df.StimDur.unique()[0] + df.Delay.unique()[0] + df.Timeout.unique()[0]
        xlim_max = df.RespWin.unique()[0] + df.StimDur.unique()[0] + df.VarDelay.unique()[0] + df.Timeout.unique()[0]

        # PLOT 5: PERISTIMULUS LICK RASTER

        time_start_raster = time.time()

        # fig = plt.figure()
        xlim = [[], []]  # Initialize empty list to store left and right xlim
        ylim = [[], []]  # Initialize empty list to store left and right ylim

        for k in range(len(df.Side.unique())):  # k=0 left trials and k=1 right trials

            if k == 0:  # Left subplot: left trials
                # ax = plt.subplot2grid((1, 2), (0, 0))
                ax = plt.subplot2grid((16, 4), (0, 0), rowspan=12, colspan=2)
                stim_color = 'tab:blue'
                ax.set_title('Left')
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
                ax.set_title('Right')
                # ax.set_xlabel('Time (s)')
                ax.set_xticklabels([])
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_yticklabels([])

            df_side = df[df.Side == k].reset_index()

            for j in range(len(df_side)):  # n trials

                # Plot stimulus length
                ax.barh(df_side.index.values[j],
                        df_side.StimLen[j],
                        left=df_side.StimStart[j] -
                             df_side.StimStart[j],
                        color=stim_color, alpha=0.5, label='Stim', zorder=1)  # Need to specify zorder otherwise
                # response window is plotted under stimulus length and can't be seen

                # Define response window color according to trial outcome
                if df_side.WrongLick[j] == 1.0:
                    resp_win_color = 'tab:pink'
                elif df_side.Hit[j] == 0.0:
                    # resp_win_color = 'tab:red'
                    resp_win_color = 'tab:gray'
                elif df_side.Hit[j] == 1.0:
                    # resp_win_color = 'tab:green'
                    resp_win_color = 'tab:gray'
                elif np.isnan(df_side.Hit[j]):
                    # resp_win_color = 'tab:gray'
                    resp_win_color = 'tab:gray'

                # # Plot delay length
                # ax.barh(df_side.index.values[j],
                #         df_side.Delay[j],
                #         left=df_side.RespWinStart[j] -
                #              df_side.StimStart[j] - df_side.Delay[j],
                #         color='lightgray', zorder=2)

                # Plot variable delay length
                ax.barh(df_side.index.values[j],
                        df_side.VarDelay[j],
                        left=df_side.RespWinStart[j] -
                             df_side.StimStart[j] - df_side.VarDelay[j],
                        color='lightgray', zorder=2)

                # Plot response window length
                ax.barh(df_side.index.values[j],
                        df_side.RespWinLen[j],
                        left=df_side.RespWinStart[j] -
                             df_side.StimStart[j],
                        color=resp_win_color, zorder=2)

                # Plot reward
                if df_side.Reward[j] == 1:
                    ax.barh(df_side.index.values[j],
                            df_side.Timeout[j],  # Same length as TimeOut as valve time is too short to be seen
                            left=df_side.RespWinEnd[j] -
                                 df_side.StimStart[j],
                            color='tab:green', alpha=0.5, zorder=2)

                # Plot timeout
                if df_side.Punish[j] == 1:
                    ax.barh(df_side.index.values[j],
                            df_side.Timeout[j],
                            left=df_side.RespWinEnd[j] -
                                 df_side.StimStart[j],
                            color='tab:red', alpha=0.5, zorder=2)

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

            xlim[k] = [ax.get_xlim()]  # Store xlim from left and right lick rasters
            # ax.set_xlim([-2, xlim[k][0][1]])  # Set xlim from -2 to trial end to zoom in and cut the fixation
            ylim[k] = [ax.get_ylim()]  # Store upper ylim from left and right lick rasters
            ax.set_xlim([-1, xlim_max])  # Set xlim from -1 to trial end to zoom in and cut the fixation
            ax.set_ylim([0, max(trials_left, trials_right)])

        # Custom legend
        legend_elements = [Patch(facecolor='tab:blue', label='Stim. left'),
                           Patch(facecolor='tab:orange', label='Stim. right'),
                           Patch(facecolor='tab:gray', label='Resp. Win.'),
                           Patch(facecolor='lightgray', label='Delay'),
                           Patch(facecolor='tab:green', label='Reward'),
                           Patch(facecolor='tab:red', label='Timeout'),
                           Patch(facecolor='tab:pink', label='WrongLick'),
                           Line2D([0], [0], marker='o', color='w', label='Left licks', markerfacecolor='tab:blue'),
                           Line2D([0], [0], marker='o', color='w', label='Right licks', markerfacecolor='tab:orange')]

        ax.legend(handles=legend_elements, loc='upper right', fontsize='xx-small', frameon=False)

        time_end_raster = time.time()
        runtime_raster = time_end_raster - time_start_raster
        print("'Plot 5: peristimulus lick raster' took", round(runtime_raster, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 6: PERISTIMULUS ALL LICKS HISTOGRAM

        time_start_psth_all = time.time()

        # fig = plt.figure()

        bin_size = 0.1

        axes_handles = []  # Store handles for plot axes as they will be overwritten by the next iteration
        ylim = [[], []]  # Initialize empty list to store left and right ylim

        for k in range(len(df.Side.unique())):  # k=0 left trials and k=1 right trials

            histcounts_L = []
            histcounts_R = []

            left_licks_per_trial = []
            right_licks_per_trial = []

            if k == 0:  # Left subplot: left trials
                # ax = plt.subplot2grid((1, 2), (0, 0))
                ax = plt.subplot2grid((16, 4), (12, 0), rowspan=2, colspan=2)
                axes_handles.append(ax)
                # ax.set_title('Left trials')
                # ax.set_xlabel('Time (s)')
                ax.set_xlim(xlim[k][0])  # Use the same xlim that left raster
                ax.set_xticklabels([])
                ax.set_ylabel('All licks\n(licks/s)')
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:  # Right subplot: right trials
                # ax = plt.subplot2grid((1, 2), (0, 1))
                ax = plt.subplot2grid((16, 4), (12, 2), rowspan=2, colspan=2)
                axes_handles.append(ax)
                # ax.set_title('Right trials')
                # ax.set_xlabel('Time (s)')
                ax.set_xlim(xlim[k][0])  # Use the same xlim that right raster
                ax.set_xticklabels([])
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_yticklabels([])

            df_side = df[df.Side == k].reset_index()

            for j in range(len(df_side)):  # n trials

                # Left licks
                for i in range(len(df_side.Port1In[j])):  # n licks
                    left_licks_per_trial.append(len(df_side.Port1In[j]))
                    if df_side.Port1In[j] == []:
                        # if not df.Port1In[j]:  # Equivalent
                        pass
                    else:
                        histcounts_L.append(df_side.Port1In[j][i] -
                                            df_side.StimStart[j])

                # Right licks
                for i in range(len(df_side.Port2In[j])):  # n licks
                    right_licks_per_trial.append(len(df_side.Port2In[j]))
                    if df_side.Port2In[j] == []:
                        # if not df.Port1In[j]:  # Equivalent
                        pass
                    else:
                        histcounts_R.append(df_side.Port2In[j][i] -
                                            df_side.StimStart[j])

            # ax.hist(histcounts_L, density=True, histtype='step', color='tab:blue', label='Left licks')
            # ax.hist(histcounts_R, density=True, histtype='step', color='tab:orange', label='Right licks')

            ylim[k] = [max(histcounts_L), max(histcounts_R)]  # Store ylims from a side

            ax.hist(histcounts_L, histtype='step', color='tab:blue', alpha=0.75, label='Left licks',
                    bins=np.linspace(-2, xlim[k][0][1]),
                    weights=np.repeat((1 / len(df[(df.Miss == 0) & (df.Side == 0)])) / bin_size,
                                      len(histcounts_L)))
            ax.hist(histcounts_R, histtype='step', color='tab:orange', alpha=0.75, label='Right licks',
                    bins=np.linspace(-2, xlim[k][0][1]),
                    weights=np.repeat((1 / len(df[(df.Miss == 0) & (df.Side == 1)])) / bin_size,
                                      len(histcounts_R)))

            # ax.set_xlim([-2, xlim[k][0][1]])  # Set xlim from -2 to trial end to zoom in and cut the fixation
            ax.set_xlim([-1, xlim_max])  # Set xlim from -1 to trial end to zoom in and cut the fixation

        # Find the maximum y-axis limit across all handles
        max_ylim = max(ax.get_ylim()[1] for ax in axes_handles)

        # Set the same maximum y-axis limit for all handles
        for ax in axes_handles:
            ax.set_ylim([0, max_ylim])

        ax.legend(loc='upper right', fontsize='xx-small', frameon=False)

        time_end_psth_all = time.time()
        runtime_psth_all = time_end_psth_all - time_start_psth_all
        print("'Plot 6: peristimulus lick histogram (all licks)' took", round(runtime_psth_all, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 7: PERISTIMULUS FIRST LICK HISTOGRAM

        time_start_psth_first = time.time()

        # fig = plt.figure()

        axes_handles = []  # Store handles for plot axes as they will be overwritten by the next iteration
        ylim = [[], []]  # Initialize empty list to store left and right ylim

        for k in range(len(df.Side.unique())):  # k=0 left trials and k=1 right trials

            first_lick_L = []
            first_lick_R = []

            if k == 0:  # Left subplot: left trials
                # ax = plt.subplot2grid((1, 2), (0, 0))
                ax = plt.subplot2grid((16, 4), (15, 0), rowspan=1, colspan=2)
                axes_handles.append(ax)
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
                axes_handles.append(ax)
                # ax.set_title('Right trials')
                ax.set_xlim([0, 2])
                ax.set_xlabel('Time (s)')
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # ax.patch.set_facecolor('none')
                ax.set_yticklabels([])

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

            ylim[k] = [max(histcounts_L), max(histcounts_R)]  # Store ylims from a side

            ax.hist(first_lick_L, histtype='step', color='tab:blue', alpha=0.75, label='Left licks',
                    bins=np.linspace(0, 2),
                    weights=np.repeat((1 / len(df[(df.Miss == 0) & (df.Side == 0)])) / bin_size,
                                      len(first_lick_L)))
            ax.hist(first_lick_R, histtype='step', color='tab:orange', alpha=0.75, label='Right licks',
                    bins=np.linspace(0, 2),
                    weights=np.repeat((1 / len(df[(df.Miss == 0) & (df.Side == 1)])) / bin_size,
                                      len(first_lick_R)))

            ax.patch.set_facecolor('none')  # Make axes transparent so the xaxes labels from the upper plot are visible
            # ax.set_xlim([-1, xlim[k][0][1]])  # Set xlim from -1 to trial end to zoom in and cut the fixation
            ax.set_xlim([-1, xlim_max])  # Set xlim from -1 to trial end to zoom in and cut the fixation

        # Find the maximum y-axis limit across all handles
        max_ylim = max(ax.get_ylim()[1] for ax in axes_handles)

        # Set the same maximum y-axis limit for all handles
        for ax in axes_handles:
            ax.set_ylim([0, max_ylim])

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
        # with open('/home/alexis/slack_bot_token', 'r') as f:  # Get slack bot token
        with open(Path.home() / 'slack_bot_token', 'r') as f:  # Get slack bot token
            slack_bot_token = f.read().replace('\n', '')

        os.environ['SLACK_BOT_TOKEN'] = slack_bot_token
        # filepath = folder + '/' + df.Session.unique()[0]
        filepath = Path(folder/df.Session.unique()[0])
        filepath = str(filepath)  # filepath, input to slack api method files.upload, used by function slack_spam,
        # requires the file path as a str
        slack_spam(msg='Hey buddy!', filepath=filepath, userid='#pv_nmdar_eranet_reports')  # Alexis: 'U01DDHH7LLX'

    ####################################################################################################################

# Line of bash code to sync the cluster data with the local machine:
# rsync -avzP -e 'ssh -p 4022' mouse@neurocomp.fcrb.es:/archive/mouse/pv_nmdar_eranet* ~/
# rsync -avzP -e 'ssh -p 4022' mouse@neurocomp.fcrb.es:/archive/mouse/pv_nmdar_eranet* ~/ && rsync -avzP -e 'ssh -p 4022' mouse@neurocomp.fcrb.es:/archive/mouse/pluginsr-for-pybpod* ~/ && rsync -avzP -e 'ssh -p 4022' mouse@neurocomp.fcrb.es:/archive/mouse/pybpod_changes* ~/

# For debugging:
# path = '/home/setup2/pv_nmdar_eranet/experiments/2AFC_5/setups/003/sessions/003_stage_training_v5_20240306-173209/003_stage_training_v5_20240306-173209.csv'
# daily_report_v5(path, send_slack=True)


# if __name__ == "__main__":
#     daily_report()

