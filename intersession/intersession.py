import time
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from matplotlib import pyplot as plt
from parse.parse import parse
from glue_sessions.glue_sessions import glue_sessions

########################################################################################################################

# Outcome:
# - Single PDF per animal, that updates after each session
# - Instead of re-doing parse of all sessions each time, save the df with all sessions and just add last one

# Objectives:
# - Track accuracy (general first, then per side)
# - Track misses (general first, then per side)
# - Track errors (general first, then per side)
# - Track bias (general first, then per side)


def intersession(animal):

    # Register time
    time_start_total = time.time()

    ####################################################################################################################

    # Import sessions

    # Group by date (not session as animals sometimes do several sessions within a day)
    # df = glue_sessions()  # '910', 'stage_training'
    path = '/home/alexis/PycharmProjects/glue_sessions/' + str(animal) + '.csv'  # Where the data for all animals is
    df = pd.read_csv(path)
    df.reset_index(drop=True, inplace=True)  # Don't create index column and modify it on the go
    # df_grouped = df.groupby('Date')  # Group by date instead of session
    sessions = df.groupby('Date').ngroup().unique()  # Array with number of dates: x axis

    ####################################################################################################################

    # Select the folder and create it if it doesn't exists
    # setup = df_session.Setup.unique()[0]  # Animal ID
    folder = '/home/alexis/Documentos/intersession reports/' #+ setup
    if not os.path.exists(folder):
        os.mkdir(folder)
    os.chdir(folder)

    ####################################################################################################################

    # Plotting style
    ms = 3  # (mpl default for plt.plot=6 and for plt.scatter=6**2)
    lw = 1.5  # mpl default

    ########################################################################################################################

    # SUMMARY VARIABLES

    # Trials
    trials = df.groupby('Date').Trial.size()
    trials_left = df[df.Side == 0].groupby('Date').Trial.size()
    trials_right = df[df.Side == 1].groupby('Date').Trial.size()

    # Hits
    hits = df.groupby('Date').Hit.sum()
    hits_left = df[df.Side == 0].groupby('Date').Hit.sum()
    hits_right = df[df.Side == 1].groupby('Date').Hit.sum()

    # Errors
    errors = df.groupby('Date').WrongLick.sum().astype(int) + df.groupby('Date').Punish.sum().astype(int)
    errors_left = df[df.Side == 0].groupby('Date').WrongLick.sum().astype(int) + \
                  df[df.Side == 0].groupby('Date').Punish.sum().astype(int)
    errors_right = df[df.Side == 1].groupby('Date').WrongLick.sum().astype(int) + \
                   df[df.Side == 1].groupby('Date').Punish.sum().astype(int)

    # Performance
    performance = hits / trials
    performance_left = hits_left / trials_left
    performance_right = hits_right / trials_right

    # Responses (valid trials)
    responses = df.groupby('Date').Response.sum()
    responses_left = df[df.Side == 0].groupby('Date').Response.sum()
    responses_right = df[df.Side == 1].groupby('Date').Response.sum()

    # Accuracy (hit rate)
    accuracy = hits / responses
    accuracy_left = hits_left / responses_left
    accuracy_right = hits_right / responses_right

    # Misses (invalid trials)
    misses = df.groupby('Date').Miss.sum()
    misses_left = df[df.Side == 0].groupby('Date').Miss.sum()
    misses_right = df[df.Side == 1].groupby('Date').Miss.sum()

    # Miss rate
    miss_rate = misses / trials
    miss_rate_left = misses_left / trials_left
    miss_rate_right = misses_right / trials_right

    # Reward
    rewards = df.groupby('Date').Reward.sum()
    rewards_left = df[df.Side == 0].groupby('Date').sum()
    rewards_right = df[df.Side == 1].groupby('Date').sum()

    # Water
    reward_size = 2.5  # μL
    water = rewards * reward_size
    water_left = rewards_left * reward_size
    water_right = rewards_right * reward_size

    ####################################################################################################################

    with PdfPages(df.Setup.unique()[0].astype(str) + '_intersession') as pdf:

        # PAGE 1

        fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches portrait
        # fig = plt.figure(figsize=(11.69, 8.27))  # A4 size in inches landscape

        ################################################################################################################

        # SUMMARY TEXT

        s1 = ('Dates: ' + df.Date.unique()[0] + ' - ' + df.Date.unique()[-1] + ', ' +
              # 'Time: ' + df.SessionStart.unique()[0][0:-7] + ' - ' + df_session.SessionEnd.unique()[0][0:-7] + ', ' +
              # [0:-7] to get rid of the floating numbers in the seconds
              'Subject: ' + df.Subject.unique()[0].astype(str) + ', ' +
              'Box: ' + df.Board.mode()[0][4] +
              '\n')

        s2 = ('Sessions: ' + str(sessions.max()))

        ################################################################################################################

        # PLOT 1: ACCURACY PER SIDE

        time_start_acc_side = time.time()

        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=1, colspan=1)

        # Plot horizontal lines
        ax1.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
        ax1.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
        ax1.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

        # Plot sides accuracy per session
        ax1.plot(sessions, accuracy, marker='o', ms=ms, lw=lw, color='black', label='Total')
        ax1.plot(sessions, accuracy_left, marker='o', ms=ms, lw=lw, color='tab:blue', label='Left')
        ax1.plot(sessions, accuracy_right, marker='o', ms=ms, lw=lw, color='tab:orange', label='Right')

        ax1.set_xlim([0, len(sessions)])
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
        # ax1.text(0, 1, s1 + s2 + s3 + s4 + s5 + s6)
        ax1.text(0, 1, s1)

        ################################################################################################################

        # PLOT 2: REPEATING VS ALTERNATING ACCURACY

        time_start_acc_repalt = time.time()

        ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=1, colspan=1)

        # Plot horizontal lines
        ax2.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
        ax2.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
        ax2.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

        rep = df[df.RepTrial == 1].groupby('Date').Hit.sum()
        alt = df[df.RepTrial == 0].groupby('Date').Hit.sum()

        # Plot rep/alt accuracy per session
        ax2.plot(sessions, accuracy_left, marker='o', ms=ms, lw=lw, color='tab:purple', label='Alt')
        ax2.plot(sessions, accuracy_right, marker='o', ms=ms, lw=lw, color='tab:brown', label='Rep')

        ax2.set_xlim([0, len(sessions)])
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

        ax3 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, colspan=1)

        # Plot horizontal lines
        ax3.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
        ax3.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
        ax3.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

        # Plot misses per session
        ax3.plot(sessions, miss_rate, marker='o', ms=ms, lw=lw, color='black', label='Total')
        ax3.plot(sessions, miss_rate_left, marker='o', ms=ms, lw=lw, color='tab:blue', label='Left')
        ax3.plot(sessions, miss_rate_right, marker='o', ms=ms, lw=lw, color='tab:orange', label='Right')

        ax3.set_xlabel('Sessions')
        ax3.set_xlim([0, len(sessions)])
        # ax3.set_xticklabels([])
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
        # ax3_twin.spines['bottom'].set_visible(False)

        time_end_miss = time.time()
        runtime_miss = time_end_miss - time_start_miss
        print("'Plot 3: misses' took", round(runtime_miss, 2), 'seconds to run')

        ################################################################################################################

        time_start_savepag1 = time.time()
        pdf.savefig()  # saves the current figure into a pdf page
        time_end_savepag1 = time.time()
        runtime_savepag1 = time_end_savepag1 - time_start_savepag1
        print("'Saving 1st page in pdf' took", round(runtime_savepag1, 2), 'seconds to run')

        plt.close()

        ################################################################################################################

    # Register time again and compute the total run time of the script
    time_end_total = time.time()
    runtime_total = time_end_total - time_start_total
    print('The script took', round(runtime_total, 2), 'seconds to run', '\n')

########################################################################################################################

# Jordi's snippets with groupby

# accu_side = df.groupby(['Side', 'Date'])['Hit'].sum()
# accu_side.loc[0]
# accu_side.loc[0].plot()
# accu_side.loc[1].plot()

# dfgroup = df.groupby(['Side', 'Date'])['Hit'].sum().reset_index()
# dfgroup.loc[dfgroup.Side==0]
# dfgroup.loc[dfgroup.Side==0, 'Hit'].plot()
