import time
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from matplotlib import pyplot as plt

from my_fun.my_fun import compute_psych_curve, slack_spam
from parse.parse import parse
from glue_sessions.glue_sessions import glue_sessions


########################################################################################################################

# Outcome:
# - Single PDF per animal, that updates after each session

# Objectives:
# - Track errors (general first, then per side)
# - Track bias (general first, then per side)

# To do:

########################################################################################################################

def intersession(path, to_csv=False, send_slack=False):
    """Do intersession reports per animal, where the x axis is the number of training days (not sessions) and y axis
    the variable of interest
    """

    # Register time
    time_start_total = time.time()

    ####################################################################################################################

    # Import dates_indexes

    # Group by date (not session as animals sometimes do several dates_indexes within a day)
    # df = glue_sessions()
    # path = '/home/alexis/PycharmProjects/glue_sessions/' + str(animal) + '.csv'  # Where the data for all animals is
    df = pd.read_csv(path)
    experiment = df.Experiment.unique()[0]
    # df.reset_index(drop=True, inplace=True)  # Don't create index column and modify it on the go
    # df_grouped = df.groupby('Date')  # Group by date instead of session
    dates_indexes = df.groupby('Date').ngroup().unique()  # Array with number of dates: x axis
    n_dates = dates_indexes.max()
    dates = df.Date.unique()

    # doi = 'yyyy-mm-dd'  # Select a date of interest to plot a vertical line
    doi_1 = '2021-05-26'  # Filename2 start being recorded
    # df.Date[df.Filename2.first_valid_index()] should return '2021-05-27'
    doi_2 = '2021-10-25'  # Messages from Arduino start being recorded
    # df.Date[np.where(df.MessageFound == 0)[0][0]] should return '2021-10-25'
    doi_3 = '2021-10-27'  # Albert board installed
    # df.Date[df.Sound.first_valid_index()] should return '2021-10-27'

    try:
        doi_1_index = np.where(dates == doi_1)[0][0]
    except IndexError:
        print(f'No data from this animal on {doi_1}')

    try:
        doi_2_index = np.where(dates == doi_2)[0][0]
    except IndexError:
        print(f'No data from this animal on {doi_2}')

    try:
        doi_3_index = np.where(dates == doi_3)[0][0]
    except IndexError:
        print(f'No data from this animal on {doi_3}')

    ####################################################################################################################

    # Select the folder where to save the PDF or create it if it doesn't exists
    setup = str(df.Setup.unique()[0])  # Animal ID
    experiment = df.Experiment.unique()[0]
    folder = '/home/alexis/Documentos/intersession reports/' + experiment  # + setup
    if not os.path.exists(folder):
        os.mkdir(folder)
    os.chdir(folder)

    ####################################################################################################################

    # Plotting style
    ms = 3  # (mpl default for plt.plot=6 and for plt.scatter=6**2)
    lw = 1.5  # mpl default

    ####################################################################################################################

    # SUMMARY VARIABLES

    # Trials
    trials = df.groupby('Date').Trial.size()
    trials_left = df[df.Side == 0].groupby('Date').Trial.size()
    trials_right = df[df.Side == 1].groupby('Date').Trial.size()

    # Hits
    hits = df.groupby('Date').Hit.sum().astype('int')
    hits_left = df[df.Side == 0].groupby('Date').Hit.sum().astype('int')
    hits_right = df[df.Side == 1].groupby('Date').Hit.sum().astype('int')
    hits_rep = df[df.RepTrial == 1].groupby('Date').Hit.sum().astype('int')  # Include in daily_report
    hits_alt = df[df.RepTrial == 0].groupby('Date').Hit.sum().astype('int')  # Include in daily_report

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

    # Repetitions/Alternations --> CHECK!!! WHY THEY'RE DIFFERENT CODED??!!
    repetitions = df[df.RepTrial == 1].groupby('Date').RepTrial.sum().astype(int)  # Include in daily_report
    alternations = df[df.RepTrial == 0].groupby('Date').RepTrial.value_counts()  # Include in daily_report

    # Accuracy (hit rate)
    accuracy = hits / responses
    accuracy_left = hits_left / responses_left
    accuracy_right = hits_right / responses_right
    accuracy_rep = hits_rep / repetitions  # Include in daily_report
    accuracy_alt = hits_alt / alternations  # Include in daily_report

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
    rewards_left = df[df.Side == 0].groupby('Date').Reward.sum()
    rewards_right = df[df.Side == 1].groupby('Date').Reward.sum()

    # Water
    reward_size = 2.5  # μL
    water = rewards * reward_size
    water_left = rewards_left * reward_size
    water_right = rewards_right * reward_size

    # Stage / Substage
    stage = df.groupby('Date').Stage.mean().round()
    # substage = df.groupby('Date').Substage.mean().round()  # Need to add it again to the parse, even if its nan
    motor = df.groupby('Date').Motor.mean().round()

    # Psychometric parameters
    # evidences = df.groupby('Date').Evidence.apply(list)  # # Need to add it again to the parse, even if its nan
    choices = df.groupby('Date').Choice.apply(list)
    psych_curves = []
    params = []
    sensitivity = []
    bias = []
    lr_left = []
    lr_right = []

    # for i in range(len(dates)):
    #     psych_curves.append(compute_psych_curve(evidences[dates[i]], choices[dates[i]]))
    #     params.append(psych_curves[i].params)
    #     sensitivity.append(params[i][0])
    #     bias.append(params[i][1])
    #     lr_left.append(params[i][2])
    #     lr_right.append(params[i][3])
    #
    # # To pandas Series
    # psych_curves = pd.Series(psych_curves, dates)
    # params = pd.Series(params, dates)
    # sensitivity = pd.Series(sensitivity, dates)
    # bias = pd.Series(bias, dates)
    # lr_left = pd.Series(lr_left, dates)
    # lr_right = pd.Series(lr_right, dates)

    # Sound
    sounds_mismatch = df['FilesMatch'].eq(0).astype(int).groupby(df['Date']).sum()
    no_sound = df['Sound'].eq(0).astype(int).groupby(df['Date']).sum()
    message_count = df.groupby('Date').MessageFound.sum()

    columns = []

    data = list(zip())

    df_intersession = pd.DataFrame(data=data, columns=columns)

    ####################################################################################################################

    with PdfPages(setup + '_intersession') as pdf:

        # PAGE 1

        fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches portrait
        # fig = plt.figure(figsize=(11.69, 8.27))  # A4 size in inches landscape

        ################################################################################################################

        # SUMMARY TEXT

        s1 = ('Dates: ' + df.Date.unique()[0] + ' - ' + df.Date.unique()[-1] + ', ' +
              'Subject: ' + df.Subject.unique()[0].astype(str) + ', ' +
              'Box: ' + df.Board.mode()[0][4] + ', ' +
              'Days: ' + str(n_dates) +
              '\n')

        ################################################################################################################

        # PLOT 0: TRIALS PER SESSION

        ax = plt.subplot2grid((8, 1), (0, 0), rowspan=1, colspan=1)

        # Plot vertical line for date of interest
        # ax.axvline(date_of_interest_index, color='tab:red', linestyle='--')

        # Plot number of trials per session
        ax.plot(dates_indexes, trials, marker='o', ms=ms, lw=lw, color='black', label='Total')
        ax.plot(dates_indexes, trials_left, marker='o', ms=ms, lw=lw, color='tab:blue', label='Left')
        ax.plot(dates_indexes, trials_right, marker='o', ms=ms, lw=lw, color='tab:orange', label='Right')

        ax.set_xlim([0, len(dates_indexes)])
        ax.set_xticklabels([])
        ax.set_ylabel('Trials')
        ax.set_ylim([0, trials.max() + 100])
        ax.set_yticks(list(np.arange(0, trials.max() + 100, 100)[0::2]))
        # ax.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        # ax.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        ax.legend(loc='lower right', fontsize='xx-small', frameon=True)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        # Instantiate a second axes that shares the same x-axis
        ax_twin = ax.twinx()
        ax_twin.set_ylim([0, trials.max() + 100])
        ax_twin.set_yticks(list(np.arange(0, trials.max() + 100, 100)[0::2]))
        ax_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        ax_twin.spines['top'].set_visible(False)
        ax_twin.spines['bottom'].set_visible(False)

        # Plot text
        # ax1.text(0, 1, s1 + s2 + s3 + s4 + s5 + s6)
        ax.text(0, ax.get_ylim()[1], s1)

        ################################################################################################################

        # PLOT 1: ACCURACY PER SIDE

        time_start_acc_side = time.time()

        ax1 = plt.subplot2grid((8, 1), (1, 0), rowspan=1, colspan=1)

        # Plot horizontal lines
        ax1.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
        ax1.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
        ax1.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

        # Plot vertical line for date of interest
        # ax1.axvline(date_of_interest_index, color='tab:red', linestyle='--')

        # Plot sides accuracy per session
        ax1.plot(dates_indexes, accuracy, marker='o', ms=ms, lw=lw, color='black', label='Total')
        ax1.plot(dates_indexes, accuracy_left, marker='o', ms=ms, lw=lw, color='tab:blue', label='Left')
        ax1.plot(dates_indexes, accuracy_right, marker='o', ms=ms, lw=lw, color='tab:orange', label='Right')

        ax1.set_xlim([0, len(dates_indexes)])
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

        ################################################################################################################

        # PLOT 2: REPEATING VS ALTERNATING ACCURACY

        time_start_acc_repalt = time.time()

        ax2 = plt.subplot2grid((8, 1), (2, 0), rowspan=1, colspan=1)

        # Plot horizontal lines
        ax2.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
        ax2.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
        ax2.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

        # Plot vertical line for date of interest
        # ax2.axvline(date_of_interest_index, color='tab:red', linestyle='--')

        # Plot rep/alt accuracy per session
        ax2.plot(dates_indexes, accuracy_alt, marker='o', ms=ms, lw=lw, color='tab:purple', label='Alt')
        ax2.plot(dates_indexes, accuracy_rep, marker='o', ms=ms, lw=lw, color='tab:brown', label='Rep')

        ax2.set_xlim([0, len(dates_indexes)])
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

        ax3 = plt.subplot2grid((8, 1), (3, 0), rowspan=1, colspan=1)

        # Plot horizontal lines
        ax3.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
        ax3.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
        ax3.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

        # Plot vertical line for date of interest
        # ax3.axvline(date_of_interest_index, color='tab:red', linestyle='--')

        # Plot misses per session
        ax3.plot(dates_indexes, miss_rate, marker='o', ms=ms, lw=lw, color='black', label='Total')
        ax3.plot(dates_indexes, miss_rate_left, marker='o', ms=ms, lw=lw, color='tab:blue', label='Left')
        ax3.plot(dates_indexes, miss_rate_right, marker='o', ms=ms, lw=lw, color='tab:orange', label='Right')

        ax3.set_xlim([0, len(dates_indexes)])
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

        # PLOT 4: STAGES/SUBSTAGES

        time_start_substages = time.time()

        ax4 = plt.subplot2grid((8, 1), (4, 0), rowspan=1, colspan=1)
        # ax4 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

        # Plot horizontal lines
        # ax4.axhline(3, color='tab:gray', linestyle=':')  # Chance level
        # ax4.axhline(6, color='tab:gray', linestyle=':')  # Accuracy 0.25
        # ax4.axhline(9, color='tab:gray', linestyle=':')  # Accuracy 0.75

        # Plot vertical line for date of interest
        # ax4.axvline(date_of_interest_index, color='tab:red', linestyle='--')

        # Plot stage/substage/motor per session
        ax4.plot(dates_indexes, stage, marker='o', ms=ms, lw=lw, color='black', label='Stage')
        # ax4.plot(dates_indexes, substage, marker='o', ms=ms, lw=lw, color='black', label='Substage')
        ax4_twin = ax4.twinx()  # Instantiate a second axes that shares the same x-axis
        ax4_twin.plot(dates_indexes, motor, marker='o', ms=ms, lw=lw, color='tab:gray', label='Motor')

        # ax4.set_xlabel('Days')
        ax4.set_xlim([0, len(dates_indexes)])
        ax4.set_xticklabels([])
        ax4.set_ylim()
        ax4.set_ylabel('Stage')
        ax4.set_yticks(stage.unique())
        # ax4.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        # ax4.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        ax4.legend(loc='lower right', fontsize='xx-small', frameon=True)
        ax4.spines['top'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        # ax4.spines['right'].set_visible(False)

        # Instantiate a second axes that shares the same x-axis
        # ax4_twin = ax4.twinx()
        # ax4_twin.set_ylim([0, 4])
        ax4_twin.set_yticks(motor.unique())
        # ax4_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        ax4_twin.set_ylabel('Motor')
        ax4_twin.spines['top'].set_visible(False)
        ax4_twin.spines['bottom'].set_visible(False)

        # Make shared legend for both axis
        lines_1, labels_1 = ax4.get_legend_handles_labels()
        lines_2, labels_2 = ax4_twin.get_legend_handles_labels()
        lines = lines_1 + lines_2
        labels = labels_1 + labels_2
        ax4.legend(lines, labels, loc='lower right', fontsize='xx-small', frameon=True)

        time_end_substages = time.time()
        runtime_substages = time_end_miss - time_start_substages
        print("'Plot 3: misses' took", round(runtime_substages, 2), 'seconds to run')

        ################################################################################################################
        # Under development
        ################################################################################################################

        # # ACCURACY PER EVIDENCE
        #
        # df.groupby(['Date', 'Evidence']).Hit.agg(['sum', 'count']).astype('int')  # This matches with daily report's accuracy when summing
        # acc_evi = df.groupby(['Date', 'Evidence']).Hit.sum() / df.groupby(['Date', 'Evidence']).Hit.count()
        #
        #
        # df.groupby(['Date', 'Evidence']).Hit.sum()['2021-10-19']

        ################################################################################################################
        # fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches portrait
        # # PLOT 5: PSYCHOMETRIC PARAMETERS
        #
        # time_start_substages = time.time()
        #
        # # ax5 = plt.subplot2grid((8, 1), (5, 0), rowspan=1, colspan=1)
        # ax5 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
        #
        # # Plot horizontal lines
        # ax5.axhline(3, color='tab:gray', linestyle=':')  # Chance level
        # ax5.axhline(6, color='tab:gray', linestyle=':')  # Accuracy 0.25
        # ax5.axhline(9, color='tab:gray', linestyle=':')  # Accuracy 0.75
        #
        # # Plot vertical line for date of interest
        # # ax5.axvline(date_of_interest_index, color='tab:red', linestyle='--')
        #
        # # Plot misses per session
        # ax5.plot(dates_indexes, sensitivity, marker='o', ms=ms, lw=lw, color='pink', label='Total')
        # ax5.plot(dates_indexes, bias, marker='o', ms=ms, lw=lw, color='olive', label='Total')
        # ax5.plot(dates_indexes, lr_left, marker='o', ms=ms, lw=lw, color='tab:blue', label='Lapse Left')
        # ax5.plot(dates_indexes, lr_right, marker='o', ms=ms, lw=lw, color='tab:orange', label='Lapse Right')
        #
        # sensitivity = pd.Series(sensitivity, dates)
        # bias = pd.Series(bias, dates)
        # lr_left = pd.Series(lr_left, dates)
        # lr_right = pd.Series(lr_right, dates)

        ################################################################################################################

        # PLOT 5: SOUND CHECKS

        # fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches portrait

        time_start_sound_checks = time.time()

        ax5 = plt.subplot2grid((8, 1), (5, 0), rowspan=1, colspan=1)

        try:
            # Plot vertical line for date of interest
            ax5.axvline(doi_1_index, color='tab:pink', linestyle='--')
        except UnboundLocalError:
            print(f'No data from this animal on {doi_1}')
        except NameError:
            print(f'No data from this animal on {doi_1}')

        try:
            # Plot vertical line for date of interest
            ax5.axvline(doi_2_index, color='tab:purple', linestyle='--')
        except UnboundLocalError:
            print(f'No data from this animal on {doi_2}')
        except NameError:
            print(f'No data from this animal on {doi_1}')

        try:
            # Plot vertical line for date of interest
            ax5.axvline(doi_3_index, color='tab:red', linestyle='--')
        except UnboundLocalError:
            print(f'No data from this animal on {doi_3}')
        except NameError:
            print(f'No data from this animal on {doi_1}')

        # # Plot sound issues per session
        ax5.plot(dates_indexes, sounds_mismatch, marker='o', ms=ms, lw=lw, color='tab:pink', label='Sounds mismatch')
        ax5.plot(dates_indexes, message_count, marker='o', ms=ms, lw=lw, color='tab:purple', label='Message count')
        ax5.plot(dates_indexes, no_sound, marker='o', ms=ms, lw=lw, color='tab:red', label='No sound')

        ax5.set_xlabel('Days')
        ax5.set_xlim([0, len(dates_indexes)])
        # ax5.set_xticklabels([])
        ax5.set_ylabel('Sound checks')
        ax5.legend(loc='upper right', fontsize='xx-small', frameon=True)
        ax5.spines['top'].set_visible(False)
        # ax5.spines['bottom'].set_visible(False)
        # ax5.spines['right'].set_visible(False)

        # Instantiate a second axes that shares the same x-axis
        ax5_twin = ax5.twinx()
        ax5_twin.set_ylim([ax5.get_ylim()[0], ax5.get_ylim()[1]])  # Get ylims from ax5 and set them for ax5_twin
        ax5_twin.spines['top'].set_visible(False)
        # ax5_twin.spines['bottom'].set_visible(False)

        time_end_sound_checks = time.time()
        runtime_sound_checks = time_end_sound_checks - time_start_sound_checks
        print("'Plot 5: sound checks' took", round(runtime_sound_checks, 2), 'seconds to run')

        ################################################################################################################

        time_start_savepag1 = time.time()
        pdf.savefig()  # saves the current figure into a pdf page
        time_end_savepag1 = time.time()
        runtime_savepag1 = time_end_savepag1 - time_start_savepag1
        print("'Saving 1st page in pdf' took", round(runtime_savepag1, 2), 'seconds to run')

        # plt.savefig(setup + '_intersession.png')  # Save as png as well
        plt.close()

        ################################################################################################################

        # Construct DataFrame
        columns = ['Dates', 'Trials', 'TrialsLeft', 'TrialsRight', 'Hits', 'HitsLeft', 'HitsRight', 'HitsRep', 'HitsAlt',
                   'Errors', 'ErrorsLeft', 'ErrorsRight', 'Performance', 'PerformanceLeft', 'PerformanceRight',
                   'Responses', 'ResponsesLeft', 'ResponsesRight', 'Repetitions', 'Alternations', 'Accuracy',
                   'AccuracyLeft', 'AccuracyRight', 'AccuracyRep', 'AccuracyAlt', 'Misses', 'MissesLeft', 'MissesRight',
                   'MissRate', 'MissRateLeft', 'MissRateRight', 'Rewards', 'RewardsLeft', 'RewardsRight', 'Water',
                   'WaterLeft', 'WaterRight', 'Stage', 'SoundsMismatch', 'NoSound', 'MessageCount']

        data = list(zip(dates, trials, trials_left, trials_right, hits, hits_left, hits_right, hits_rep, hits_alt, errors,
                        errors_left, errors_right, performance, performance_left, performance_right, responses,
                        responses_left, responses_right, repetitions, alternations, accuracy, accuracy_left,
                        accuracy_right, accuracy_rep, accuracy_alt, misses, misses_left, misses_right, miss_rate,
                        miss_rate_left, miss_rate_right, rewards, rewards_left, rewards_right, water, water_left,
                        water_right, stage, sounds_mismatch, no_sound, message_count))

        df_intersession = pd.DataFrame(data=data, columns=columns)

    if to_csv:
        df_intersession.to_csv(folder + '/' + setup + '_intersession.csv', index=False)  # index=False to avoid the 'Unmmaed: 0' column

    # Register time again and compute the total run time of the script
    time_end_total = time.time()
    runtime_total = time_end_total - time_start_total
    print('The script took', round(runtime_total, 2), 'seconds to run', '\n')

    # This block needs to be the last otherwise it sends the file too soon and corrupted
    if send_slack:
        with open('/home/alexis/slack_bot_token', 'r') as f:  # Get slack bot token
            slack_bot_token = f.read().replace('\n', '')

        os.environ['SLACK_BOT_TOKEN'] = slack_bot_token
        # filepath = folder + '/' + df.Session.unique()[0]
        filepath = folder + '/' + setup + '_intersession'
        slack_spam(msg='Hey buddy!', filepath=filepath, userid='#pv_nmdar_eranet_reports')  # Alexis: 'U01DDHH7LLX'

    return df_intersession


########################################################################################################################

def do_intersessions(experiment=None, to_csv=False, send_slack=False):
    """Do the intersessions for all animals of a given batch (experiment)"""

    time_start = time.time()

    # To do:
    # Update glue_sessions before

    if experiment is None:

        folder = '/home/alexis/pv_nmdar_eranet/experiments/'  # Where the data for all animals is
        experiments = os.listdir(folder)  # List experiments
        experiments.sort()  # Sort them by name

        try:
            experiments.remove('.idea')  # Pycharm's archive
            experiments.remove('Daily check')
            experiments.remove('WaterCalibration')
        except ValueError:
            pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    # print('Doing intersession reports of: ' + animal)
    folder = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'
    # animal = input('Enter animal')
    animals = os.listdir(folder)
    animals = [animals for animals in animals if animals.endswith('.csv') and len(animals) == 7]
    animals.sort()

    for i in range(len(animals)):
        path = folder + animals[i]
        intersession(path, to_csv=to_csv, send_slack=send_slack)

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')


########################################################################################################################

def learning_trajectories(experiment=None):

    time_start = time.time()

    if experiment is None:
        folder = folder = '/home/alexis/Documentos/intersession reports/'  # Where the data for all animals is
        experiments = os.listdir(folder)  # List experiments
        experiments.sort()  # Sort them by name
        experiments = [x for x in experiments if os.path.isdir(folder + x)]  # Get rid of non folders
        # experiments = next(os.walk(folder))[1]  # Same but with another method
        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    folder = folder = '/home/alexis/Documentos/intersession reports/' + experiment + '/'

    animals = os.listdir(folder)
    animals = [animals for animals in animals if animals.endswith('.csv')]
    animals.sort()

    df_all_intersessions = pd.DataFrame()  # Create empty DataFrame
    plt.figure()

    for i in range(len(animals)):
        path = folder + animals[i]
        df = pd.read_csv(path)
        df_all_intersessions = pd.concat([df_all_intersessions, df])
        plt.plot(df.index, df.Accuracy, color='tab:gray', alpha=0.5)

    # Hits
    hits = df_all_intersessions.groupby('Dates').Hits.sum().astype('int')
    hits_left = df_all_intersessions.groupby('Dates').HitsLeft.sum().astype('int')
    hits_right = df_all_intersessions.groupby('Dates').HitsRight.sum().astype('int')

    # Responses (valid trials)
    responses = df_all_intersessions.groupby('Dates').Responses.sum()
    responses_left = df_all_intersessions.groupby('Dates').ResponsesLeft.sum()
    responses_right = df_all_intersessions.groupby('Dates').ResponsesRight.sum()

    # Accuracy (hit rate)
    accuracy = hits / responses
    accuracy_left = hits_left / responses_left
    accuracy_right = hits_right / responses_right

    plt.plot(np.arange(0, len(accuracy)), accuracy, color='k', linewidth=3)

    plt.title('Learning trajectories')
    plt.xlabel('Days')
    plt.ylabel('Accuracy')

    plt.savefig(folder + experiment + '_learning_trajectories.png')

########################################################################################################################

# Jordi's snippets with groupby:

# accu_side = df.groupby(['Side', 'Date'])['Hit'].sum()
# accu_side.loc[0]
# accu_side.loc[0].plot()
# accu_side.loc[1].plot()

# dfgroup = df.groupby(['Side', 'Date'])['Hit'].sum().reset_index()
# dfgroup.loc[dfgroup.Side==0]
# dfgroup.loc[dfgroup.Side==0, 'Hit'].plot()

# s.groupby(['prob_repeat', 'aftererror'])['aftererror','hithistory'].agg(['mean', 'count', np.std, scipy.stats.sem])
