import time
from pathlib import Path
import pandas as pd
import datetime
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from glue_sessions.glue_sessions import update_glued_sessions
from my_fun.my_fun import *
# import statsmodels.formula.api as smf


########################################################################################################################

# Objectives:
# - Track errors (general first, then per side)
# - Track bias (general first, then per side)

# To do:

########################################################################################################################

@timer
def intersession_within_animal(path, alignment='n_sessions', to_csv=False, send_slack=False):
    """Do intersession report per animal, where the x axis is the number of training days (not sessions) and y axis
    the variable of interest
    """

    # Import dates_indexes
    # Group by date (not session as animals sometimes do several dates_indexes within a day)
    # df = glue_sessions()
    # path = '/home/alexis/PycharmProjects//' + str(animal) + '.csv'  # Where the data for all animals is
    # path = Path.home() / 'PycharmProjects' / 'glue_sessions' / '2AFC_X' / str(animal).csv  # Where the data for all animals is
    df = pd.read_csv(path)
    experiment = df.Experiment.unique()[0]
    # setup = str(df.Setup.unique()[0])  # Animal ID  # This returns an array in the case that I trained the wrong mouse
    # by mistake in another box
    setup = str(df.Setup.mode()[0])  # Animal ID
    df = df[df.Setup == int(setup)]  # Exclude those sessions with other mice by mistake
    print(f'Doing the intersession of animal {setup} from experiment {experiment}...')
    # df.reset_index(drop=True, inplace=True)  # Don't create index column and modify it on the go
    # df_grouped = df.groupby('Date')  # Group by date instead of session

    # dates_indexes = df.groupby('Date').ngroup().unique()  # Array with number of dates: x-axis
    # n_dates = dates_indexes.max()

    dates = df.Date.dropna().unique()  # Dropna because there's some corrupted trials in which the date is nan
    dates_indexes = np.arange(len(dates))
    n_dates = len(dates)
    dow = [datetime.datetime.strptime(dates[i], "%Y-%m-%d").date().weekday() for i in range(len(dates))]  # Date of the
    # week, Monday is 0 and Sunday is 6
    dates_datetime = [datetime.datetime.strptime(dates[i], "%Y-%m-%d") for i in range(len(dates))]

    ####################################################################################################################

    # Import date of birth (to align plots to DOB if desired)
    try:
        dob = pd.read_csv(Path.home() / 'PycharmProjects' / 'intersession' / 'DOB.csv')  # From Mice's - Overview Google
        # Sheets (https://docs.google.com/spreadsheets/d/1hNnBMbe4se3VPOn5FeS1ViTVfXVZ7YLJCiFRWFPYMoU/edit#gid=551555314)
        dob_datetime = []
        for i in range(len(dob)):
            try:
                dob_datetime.append(datetime.datetime.strptime(dob.DOB[i], "%d/%m/%Y"))
            except ValueError:  # time data 'Sep/Aug 2020' does not match format '%d/%m/%Y'
                dob_datetime.append(np.nan)
            except TypeError:  # strptime() argument 1 must be str, not float
                dob_datetime.append(np.nan)
        dob['DOB_datetime'] = dob_datetime
        dob_current_mouse = dob[dob.ID == setup]['DOB_datetime'].iloc[-1]
        age = [(dates_datetime[i] - dob_current_mouse).days for i in range(len(dates_datetime))]
        aoi = np.arange(0, 365, 30)
        aoi[(aoi > age[0]) & (aoi < age[-1])]
        # aoi = np.arange(age[0], age[-1], 30)  # Age of interest (every 30 days)
    except IndexError:
        print('Could not recover DOB data, filling with NaNs instead')
        age = n_dates * [np.nan]
    ####################################################################################################################

    # Dates of interest
    # doi = 'yyyy-mm-dd'  # Select a date of interest to plot a vertical line

    # For 2AFC_2 (batch 2)
    doi_0 = '2021-05-26'  # Filename2 start being recorded
    # df.Date[df.Filename2.first_valid_index()] should return '2021-05-27'
    doi_1 = '2021-10-25'  # Messages from Arduino start being recorded
    # df.Date[np.where(df.MessageFound == 0)[0][0]] should return '2021-10-25'
    doi_2 = '2021-10-27'  # Albert board installed
    # df.Date[df.Sound.first_valid_index()] should return '2021-10-27'
    doi_3 = '2022-02-21'  # First training day after the retreat (February 17-19)
    doi_4 = '2022-02-25'  # Pep visited the animals in sala C to examine 326 and 329
    doi_5 = '2022-03-03'  # First day without bringing down Tiffany's animals
    doi_6 = '2022-03-09'  # Mice moved from sala C to sala B
    doi_7 = '2022-03-21'  # Ad lib CA 2% water
    doi_8 = '2022-04-07'  # First day without bringing down Balma's animals
    # Add elastic p (March 11)

    # For 2AFC_3 (batch 3)
    doi_9 = '2022-09-12'  # First session after BAMB! 2022
    doi_10 = '2022-09-13'  # Ad lib. CA 2% in the cage + 10% sweetened water in the task
    doi_11 = '2022-10-10'  # Caged alone

    # For 2AFC_4 (batch 4)
    doi_12 = '2023-03-08'  # Introduction of blocks
    doi_13 = '2023-03-13'  # Removal of blocks
    doi_14 = '2023-03-20'  # Fixed motor coming after StimulusDuration and not at the end of Delay
    doi_15 = '2023-03-24'  # Removed motor in AW
    doi_16 = '2023-03-30'  # Reintroduction of blocks
    doi_17 = '2023-03-31'  # Added task parameter to choose from a reaction time (RT) or a fixed duration (FD)
    doi_18 = '2023-05-09'  # Added variable delay
    doi_19 = '2023-05-10'  # Installation of SAI on setup PCs and of industrial quality SD cards
    doi_20 = '2023-05-15'  # Installation of powered USB hubs
    doi_21 = '2023-07-10'  # First session after ERANET meeting 2023

    # For 2AFC_5 (batch 5)
    doi_22 = '2024-03-26'  # Change of sounds in SD card: 5*100 ms frames, 60 dB
    doi_23 = '2024-04-02'  # First session after Easter Break (3 days without training with 0% ad lib. water)

    # For 2AFC_6 (batch 6)
    doi_24 = '2025-06-26'  # First attempt ti fix XYZ coordinates per mouse
    doi_25 = '2025-07-15'  # Drug experiment beginning

    # For testing specific events
    # doi_test = '2023-06-09'
    # dois = [doi_test]

    dois = [doi_0, doi_1, doi_2, doi_3, doi_4, doi_5, doi_6, doi_7, doi_8,  # Batch 2
            doi_9, doi_10, doi_11,  # Batch 3
            doi_14, doi_15, doi_17, doi_18, doi_19, doi_20, doi_21,
            # Batch 4  (skipped doi_12, doi_13, doi_16 for clarity)
            doi_22, doi_23,  # Batch 5
            doi_24, doi_25]
    dois_indexes = []

    for i in range(len(dois)):
        try:
            dois_indexes.append(np.where(dates == dois[i])[0][0])
        except IndexError:
            print(f'No data from this animal on {dois[i]}')
            dois_indexes.append(np.nan)

    ####################################################################################################################

    # Alignment (x-axis)
    if alignment == 'n_sessions':  # Number of training sessions. Starts at 0
        x = dates_indexes
        xlim = [0, len(x)]
        xlabel = 'N sessions'
    elif alignment == 'age':  # Age of the animal in days. Doesn't start at 0
        x = age
        xlim = [min(age), max(age)]
        xlabel = 'Age (days)'

    ####################################################################################################################

    # Select the folder where to save the PDF or create it if it doesn't exist
    folder_pdf_out = Path.home() / 'Documents' / 'intersession reports' / experiment

    if not os.path.exists(folder_pdf_out):
        # os.mkdir(folder_pdf_out)
        folder_pdf_out.mkdir(parents=True, exist_ok=True)
    os.chdir(folder_pdf_out)

    ####################################################################################################################

    # Plotting style
    ms = 3  # (mpl default for plt.plot=6 and for plt.scatter=6**2)
    lw = 1.5  # mpl default

    ####################################################################################################################

    # SUMMARY VARIABLES

    subject = df.groupby('Date').Subject.unique().astype('int')
    board = df.groupby('Date').Board.transform(lambda x: x.unique()[0])

    # Trials
    trials = df.groupby('Date').Trial.size()
    trials_left = df[df.Side == 0].groupby('Date').Trial.size()
    trials_right = df[df.Side == 1].groupby('Date').Trial.size()

    # Choice
    chose_left = df[df.Choice == 0].groupby('Date').size()
    chose_right = df[df.Choice == 1].groupby('Date').size()

    # Hits
    hits = df.groupby('Date').Hit.sum().astype('int')
    hits_left = df[df.Side == 0].groupby('Date').Hit.sum().astype('int')
    hits_right = df[df.Side == 1].groupby('Date').Hit.sum().astype('int')
    hits_rep = df[df.RepTrial == 1].groupby('Date').Hit.sum().astype('int')  # Include in daily_report
    hits_alt = df[df.RepTrial == 0].groupby('Date').Hit.sum().astype('int')  # Include in daily_report
    hits_max_evi = df[(df.ILD == df.ILD.min()) | (df.ILD == df.ILD.max())].groupby('Date').Hit.sum().astype('int')

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
    responses_max_evi = df[(df.ILD == df.ILD.min()) | (df.ILD == df.ILD.max())].groupby('Date').Response.sum()

    # Response rate
    response_rate = responses / trials
    response_rate_left = responses_left / trials_left
    response_rate_right = responses_right / trials_right

    # Repetitions/Alternations
    repetitions = df[df.RepChoice == 1].groupby('Date').RepChoice.sum().astype(int)  # Include in daily_report
    reps_left = df[df.Choice == 0].groupby('Date').RepChoice.sum().astype('int')
    reps_right = df[df.Choice == 1].groupby('Date').RepChoice.sum().astype('int')
    rep_rate_left = reps_left / chose_left
    rep_rate_right = reps_right / chose_right
    alternations = df[df.RepChoice == 0].groupby('Date').RepChoice.size()  # Include in daily_report
    alts_left = df[(df.RepChoice == 0) & (df.Choice == 0)].groupby('Date').RepChoice.size()
    alts_right = df[(df.RepChoice == 0) & (df.Choice == 1)].groupby('Date').RepChoice.size()
    alt_rate_left = alts_left / chose_left
    alt_rate_right = alts_right / chose_right

    # Accuracy (hit rate)
    accuracy = hits / responses
    accuracy_left = hits_left / responses_left
    accuracy_right = hits_right / responses_right
    accuracy_max_evi = hits_max_evi / responses_max_evi
    lateral_bias = accuracy_right - accuracy_left
    accuracy_rep = hits_rep / repetitions  # Include in daily_report
    accuracy_alt = hits_alt / alternations  # Include in daily_report
    rep_bias = accuracy_rep - accuracy_alt
    corr_rep_bias = rep_rate_left * 0.5 + rep_rate_right * 0.5  # Corrected for the lateral bias. Equivalent to the mean
    # but like this the output is a pd.Series
    corr_alt_bias = alt_rate_left * 0.5 + alt_rate_right * 0.5

    # Delay
    try:
        var_delay = df.groupby('Date').VarDelay.mean()
    except AttributeError:  # 'DataFrameGroupBy' object has no attribute 'VarDelay'
        var_delay = [np.nan] * len(dates)

    # Blocks were introduced in batch 4, so this won't work for the first 3 batches
    try:
        # Accuracy blocks (accuracy of first trial of each block)
        blocks = df.groupby('Date').Blocks.unique()
        # block_len = df.groupby('Date').BlockLen.unique()
        side = df.groupby('Date').Side.apply(list)
        warm_up = df.groupby('Date').WarmUp.unique()
        block_change_indexes = []
        block_change_dist = []  # Should be the same as block_len, but block_len wasn't from the beginning of blocks
        block_change_dist_mode = []
        accuracy_blocks = []
        accuracy_blocks_left = []
        accuracy_blocks_right = []

        for i in range(len(dates)):
            df_session = df[df.Date == dates[i]].reset_index()
            # Take only sessions with blocks (VAR_BLOCKS not nan and != 0) and without warm_up (VAR_WARM_UP == 0) as warm up
            # is used to transition from blocks to random
            if not pd.isnull(blocks[i][0]) and blocks[i][0] != '0' and warm_up[i][0] == 0:
                block_change_indexes.append([j for j in range(1, len(side[i])) if side[i][j - 1] != side[i][j]])
                block_change_dist.append([block_change_indexes[i][j] - block_change_indexes[i][j - 1] for j in
                                          range(1, len(block_change_indexes[i]))])
                block_change_dist_mode.append(float(max(set(block_change_dist[i]), key=block_change_dist[i].count)))
                accuracy_blocks.append(df_session.loc[block_change_indexes[i]].Hit.mean())
                accuracy_blocks_left.append(
                    df_session.loc[block_change_indexes[i]].loc[df_session.Side == 0].Hit.mean())
                accuracy_blocks_right.append(
                    df_session.loc[block_change_indexes[i]].loc[df_session.Side == 1].Hit.mean())
            else:
                block_change_indexes.append(np.nan)
                block_change_dist.append(np.nan)
                block_change_dist_mode.append(np.nan)
                accuracy_blocks.append(np.nan)
                accuracy_blocks_left.append(np.nan)
                accuracy_blocks_right.append(np.nan)
    except AttributeError:
        pass

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

    # Sound
    sounds_mismatch = df['FilesMatch'].eq(0).astype(int).groupby(df['Date']).sum()
    no_sound = df['Sound'].eq(0).astype(int).groupby(df['Date']).sum()
    message_count = df.groupby('Date').MessageFound.sum()

    # Probabilities of difficult trials (non-maximum evidence)
    p = df.groupby('Date').P.mean()
    p = p.fillna(0)

    # Psychometric parameters
    # evidences = df.groupby('Date').Evidence.apply(list)  # Need to add it again to the parse, even if its nan
    ilds = df.groupby('Date').ILD.apply(list)
    ilds_rep = df.groupby('Date').ILDRep.apply(list)
    choices = df.groupby('Date').Choice.apply(list)
    rep_choices = df.groupby('Date').RepChoice.apply(list)

    # Prob. right
    pc_right = []
    params_pc_right = []
    sensitivity_pc_right = []
    bias_pc_right = []
    lapse_right = []
    lapse_left = []

    xdata_pc_right = []
    ydata_pc_right = []
    fit_pc_right = []
    fit_error_pc_right = []

    # Prob. rep
    pc_rep = []
    params_pc_rep = []
    sensitivity_pc_rep = []
    bias_pc_rep = []
    lapse_rep = []
    lapse_alt = []

    xdata_pc_rep = []
    ydata_pc_rep = []
    fit_pc_rep = []
    fit_error_pc_rep = []

    for i in range(len(dates)):
        # Prob. right
        pc_right.append(compute_psych_curve(ilds[dates[i]], choices[dates[i]]))
        params_pc_right.append(pc_right[i].params)
        sensitivity_pc_right.append(params_pc_right[i][0])
        bias_pc_right.append(params_pc_right[i][1])
        lapse_right.append(params_pc_right[i][2])
        lapse_left.append(params_pc_right[i][3])

        xdata_pc_right.append(pc_right[i].xdata)
        ydata_pc_right.append(pc_right[i].ydata)
        fit_pc_right.append(pc_right[i].fit)
        fit_error_pc_right.append(pc_right[i].fit_error)

        # Prob. rep
        pc_rep.append(compute_psych_curve(ilds_rep[dates[i]], rep_choices[dates[i]]))
        params_pc_rep.append(pc_rep[i].params)
        sensitivity_pc_rep.append(params_pc_rep[i][0])
        bias_pc_rep.append(params_pc_rep[i][1])
        lapse_rep.append(params_pc_rep[i][2])
        lapse_alt.append(params_pc_rep[i][3])

        xdata_pc_rep.append(pc_right[i].xdata)
        ydata_pc_rep.append(pc_right[i].ydata)
        fit_pc_rep.append(pc_right[i].fit)
        fit_error_pc_rep.append(pc_right[i].fit_error)

    # To pandas Series
    # Prob. right
    pc_right = pd.Series(pc_right, dates)
    params_pc_right = pd.Series(params_pc_right, dates)
    sensitivity_pc_right = pd.Series(sensitivity_pc_right, dates)
    bias_pc_right = pd.Series(bias_pc_right, dates)
    lapse_left = pd.Series(lapse_left, dates)
    lapse_right = pd.Series(lapse_right, dates)

    xdata_pc_right = pd.Series(xdata_pc_right, dates)
    ydata_pc_right = pd.Series(ydata_pc_right, dates)
    fit_pc_right = pd.Series(fit_pc_right, dates)
    fit_error_pc_right = pd.Series(fit_error_pc_right, dates)

    # Prob. rep
    pc_rep = pd.Series(pc_rep, dates)
    params_pc_rep = pd.Series(params_pc_rep, dates)
    sensitivity_pc_rep = pd.Series(sensitivity_pc_rep, dates)
    bias_pc_rep = pd.Series(bias_pc_rep, dates)
    lapse_alt = pd.Series(lapse_alt, dates)
    lapse_rep = pd.Series(lapse_rep, dates)

    xdata_pc_rep = pd.Series(xdata_pc_rep, dates)
    ydata_pc_rep = pd.Series(ydata_pc_rep, dates)
    fit_pc_rep = pd.Series(fit_pc_rep, dates)
    fit_error_pc_rep = pd.Series(fit_error_pc_rep, dates)

    # Drug
    drug = df.groupby('Date').Drug.unique().astype(float)

    ####################################################################################################################

    filename = f'{setup}_intersession_({alignment}_aligned).pdf'

    with PdfPages(filename) as pdf:

        # PAGE 1

        fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches portrait
        # fig = plt.figure(figsize=(11.69, 8.27))  # A4 size in inches landscape

        ################################################################################################################

        # SUMMARY TEXT
        new_line = '\n'  # Trick to include new lines in formatted strings
        # https://towardsdatascience.com/how-to-add-new-line-in-python-f-strings-7b4ccc605f4a
        sum_text = (f'Dates: {df.Date.unique()[0]} - {df.Date.unique()[-1]}, '
                    f'Subject: {df.Subject.unique()[0].astype(str)}, '
                    f'Box: {df.Board.mode()[0][4]}, '
                    f'Days: {str(n_dates)}'
                    f'{new_line}'
                    f'{new_line}')

        ################################################################################################################

        # PLOT 0: ACCURACY PER SIDE

        time_start = time.time()

        ax = plt.subplot2grid((9, 1), (0, 0), rowspan=1, colspan=1)

        # Plot horizontal lines
        ax.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
        ax.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
        ax.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

        # Plot vertical line for date of interest
        if alignment == 'n_sessions':
            for i in range(len(dois)):
                try:
                    ax.axvline(dois_indexes[i], color='tab:red', linestyle='--')
                except IndexError:
                    pass
        elif alignment == 'age':
            for i in range(len(aoi)):
                ax.axvline(aoi[i], color='tab:red', linestyle='--')

        # Plot sides accuracy per session
        ax.plot(x, accuracy, marker='o', ms=ms, lw=lw, color='black', label='Total')
        ax.plot(x, accuracy_left, marker='o', ms=ms, lw=lw, color='tab:blue', label='Left')
        ax.plot(x, accuracy_right, marker='o', ms=ms, lw=lw, color='tab:orange', label='Right')

        ax.set_xlim(xlim)
        ax.set_xticklabels([])
        ax.set_ylabel('Acc.\nL/R (%)')
        ax.set_ylim([0, 1.1])
        ax.set_yticks(list(np.arange(0, 1.1, 0.1)))
        ax.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        # ax.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        ax.legend(loc='lower right', fontsize='xx-small', frameon=True)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        # Instantiate a second axes that shares the same x-axis
        ax_twin = ax.twinx()
        ax_twin.set_ylim([0, 1.1])
        ax_twin.set_yticks(list(np.arange(0, 1.1, 0.1)))
        ax_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        ax_twin.spines['top'].set_visible(False)
        ax_twin.spines['bottom'].set_visible(False)

        time_end = time.time()
        runtime = time_end - time_start
        print("'Plot 0: 'accuracy per sides' took", round(runtime, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 1: LATERAL BIAS

        time_start = time.time()

        ax1 = plt.subplot2grid((9, 1), (1, 0), rowspan=1, colspan=1)

        # Plot horizontal lines
        ax1.axhline(0, color='tab:gray', linestyle='--')  # Unbias
        ax1.axhline(-0.5, color='tab:gray', linestyle=':')  # Bias to the left
        ax1.axhline(0.5, color='tab:gray', linestyle=':')  # Bias to the right

        # Plot vertical line for date of interest
        if alignment == 'n_sessions':
            for i in range(len(dois)):
                try:
                    ax1.axvline(dois_indexes[i], color='tab:red', linestyle='--')
                except IndexError:
                    pass
        elif alignment == 'age':
            for i in range(len(aoi)):
                ax1.axvline(aoi[i], color='tab:red', linestyle='--')

        # Plot lateral bias per session
        ax1.plot(x, lateral_bias, marker='o', ms=ms, lw=lw, color='black')

        ax1.set_xlim(xlim)
        ax1.set_xticklabels([])
        ax1.set_ylabel('Lateral\nBias (%)')
        ax1.set_ylim([-1, 1])
        ax1.set_yticks(list(np.linspace(-1, 1, 11)))
        ax1.set_yticklabels(['L', '', '', '', '', '0', '', '', '', '', 'R'])
        # ax1.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        # ax1.legend(loc='lower right', fontsize='xx-small', frameon=True)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        # ax1.spines['right'].set_visible(False)

        # Instantiate a second axes that shares the same x-axis
        ax1_twin = ax1.twinx()
        ax1_twin.set_ylim([-1, 1])
        ax1_twin.set_yticks(list(np.linspace(-1, 1, 11)))
        ax1_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        ax1_twin.spines['top'].set_visible(False)
        ax1_twin.spines['bottom'].set_visible(False)

        time_end = time.time()
        runtime = time_end - time_start
        print("'Plot 1: 'lateral bias' took", round(runtime, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 2: REPEATING VS ALTERNATING ACCURACY

        time_start = time.time()

        ax2 = plt.subplot2grid((9, 1), (2, 0), rowspan=1, colspan=1)

        # Plot horizontal lines
        ax2.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
        ax2.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
        ax2.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

        # Plot vertical line for date of interest
        if alignment == 'n_sessions':
            for i in range(len(dois)):
                try:
                    ax2.axvline(dois_indexes[i], color='tab:red', linestyle='--')
                except IndexError:
                    pass
        elif alignment == 'age':
            for i in range(len(aoi)):
                ax2.axvline(aoi[i], color='tab:red', linestyle='--')

        # Plot rep/alt accuracy per session
        ax2.plot(x, accuracy_alt, marker='o', ms=ms, lw=lw, color='tab:purple', label='Alt')
        ax2.plot(x, accuracy_rep, marker='o', ms=ms, lw=lw, color='tab:brown', label='Rep')

        ax2.set_xlim(xlim)
        ax2.set_xticklabels([])
        ax2.set_ylabel('Acc.\nAlt/Rep (%)')
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

        time_end = time.time()
        runtime = time_end - time_start
        print("'Plot 2: 'accuracy repeating vs alternating' took", round(runtime, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 3: BLOCK ACCURACY

        time_start = time.time()

        ax3 = plt.subplot2grid((9, 1), (3, 0), rowspan=1, colspan=1)

        # Plot horizontal lines
        ax3.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
        ax3.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
        ax3.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

        # Plot vertical line for date of interest
        if alignment == 'n_sessions':
            for i in range(len(dois)):
                try:
                    ax3.axvline(dois_indexes[i], color='tab:red', linestyle='--')
                except IndexError:
                    pass
        elif alignment == 'age':
            for i in range(len(aoi)):
                ax3.axvline(aoi[i], color='tab:red', linestyle='--')

        # Blocks were introduced in batch 4, so this won't work for the first 3 batches
        try:
            # Plot block accuracy per session
            ax3.plot(x, accuracy_blocks, marker='o', ms=ms, lw=lw, color='black', label='Total')
            ax3.plot(x, accuracy_blocks_left, marker='o', ms=ms, lw=lw, color='tab:blue', label='Left')
            ax3.plot(x, accuracy_blocks_right, marker='o', ms=ms, lw=lw, color='tab:orange', label='Right')
        except UnboundLocalError:
            pass

        ax3.set_xlim(xlim)
        ax3.set_xticklabels([])
        ax3.set_ylabel('Acc.\nblocks (%)')
        ax3.set_ylim([0, 1.1])
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

        time_end = time.time()
        runtime = time_end - time_start
        print("'Plot 3: 'block accuracy' took", round(runtime, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 4: BLOCK LENGTH

        time_start = time.time()

        ax4 = plt.subplot2grid((9, 1), (4, 0), rowspan=1, colspan=1)

        # Plot vertical line for date of interest
        if alignment == 'n_sessions':
            for i in range(len(dois)):
                try:
                    ax4.axvline(dois_indexes[i], color='tab:red', linestyle='--')
                except IndexError:
                    pass
        elif alignment == 'age':
            for i in range(len(aoi)):
                ax4.axvline(aoi[i], color='tab:red', linestyle='--')

        # Blocks were introduced in batch 4, so this won't work for the first 3 batches
        try:
            # Plot block_length per session
            ax4.plot(x, block_change_dist_mode, marker='o', ms=ms, lw=lw, color='black')
        except UnboundLocalError:
            pass

        # ax4.set_xlabel('Days')
        ax4.set_xlim(xlim)
        ax4.set_xticklabels([])
        # ax4.xaxis.get_major_locator().set_params(integer=True)  # Force integers only in x ticks
        ax4.set_ylabel('Block length')
        # ax4.legend(loc='upper right', fontsize='xx-small', frameon=True)
        ax4.spines['top'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        # ax4.spines['right'].set_visible(False)

        # Instantiate a second axes that shares the same x-axis
        ax4_twin = ax4.twinx()
        ax4_twin.set_ylim([ax4.get_ylim()[0], ax4.get_ylim()[1]])  # Get ylims from ax6 and set them for ax6_twin
        ax4_twin.set_yticklabels([])
        ax4_twin.spines['top'].set_visible(False)
        ax4_twin.spines['bottom'].set_visible(False)

        time_end = time.time()
        runtime = time_end - time_start
        print("'Plot 6: 'block length' took", round(runtime, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 5: MISSES

        time_start = time.time()

        ax5 = plt.subplot2grid((9, 1), (5, 0), rowspan=1, colspan=1)

        # Plot horizontal lines
        ax5.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
        ax5.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
        ax5.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

        # Plot vertical line for date of interest
        if alignment == 'n_sessions':
            for i in range(len(dois)):
                try:
                    ax5.axvline(dois_indexes[i], color='tab:red', linestyle='--')
                except IndexError:
                    pass
        elif alignment == 'age':
            for i in range(len(aoi)):
                ax5.axvline(aoi[i], color='tab:red', linestyle='--')

        # Plot misses per session
        ax5.plot(x, miss_rate, marker='o', ms=ms, lw=lw, color='black', label='Total')
        ax5.plot(x, miss_rate_left, marker='o', ms=ms, lw=lw, color='tab:blue', label='Left')
        ax5.plot(x, miss_rate_right, marker='o', ms=ms, lw=lw, color='tab:orange', label='Right')

        ax5.set_xlim(xlim)
        ax5.set_xticklabels([])
        ax5.set_ylim([0, 1.1])
        ax5.set_ylabel('Miss rate\nL/R (%)')
        ax5.set_yticks(list(np.arange(0, 1.1, 0.1)))
        ax5.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        # ax5.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        ax5.legend(loc='lower right', fontsize='xx-small', frameon=True)
        ax5.spines['top'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        # ax5.spines['right'].set_visible(False)

        # Instantiate a second axes that shares the same x-axis
        ax5_twin = ax5.twinx()
        ax5_twin.set_ylim([0, 1.1])
        ax5_twin.set_yticks(list(np.arange(0, 1.1, 0.1)))
        ax5_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        ax5_twin.spines['top'].set_visible(False)
        ax5_twin.spines['bottom'].set_visible(False)

        time_end = time.time()
        runtime = time_end - time_start
        print("'Plot 2: 'misses' took", round(runtime, 2), 'seconds to run')

        ################################################################################################################

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
        # if alignment == 'n_sessions':
        #     for i in range(len(dois)):
        #         try:
        #             ax5.axvline(dois_indexes[i], color='tab:red', linestyle='--')
        #         except IndexError:
        #             pass
        # elif alignment == 'age':
        #     for i in range(len(aoi)):
        #         ax5.axvline(aoi[i], color='tab:red', linestyle='--')
        #
        # # Plot misses per session
        # ax5.plot(x, sensitivity_pc_right, marker='o', ms=ms, lw=lw, color='pink', label='Total')
        # ax5.plot(x, bias, marker='o', ms=ms, lw=lw, color='olive', label='Total')
        # ax5.plot(x, lapse_left, marker='o', ms=ms, lw=lw, color='tab:blue', label='Lapse Left')
        # ax5.plot(x, lapse_right, marker='o', ms=ms, lw=lw, color='tab:orange', label='Lapse Right')
        #
        # sensitivity_pc_right = pd.Series(sensitivity_pc_right, dates)
        # bias = pd.Series(bias, dates)
        # lapse_left = pd.Series(lapse_left, dates)
        # lapse_right = pd.Series(lapse_right, dates)

        ################################################################################################################

        # PLOT 6: SOUND CHECKS

        time_start = time.time()

        ax6 = plt.subplot2grid((9, 1), (6, 0), rowspan=1, colspan=1)

        # Plot vertical line for date of interest
        if alignment == 'n_sessions':
            for i in range(len(dois)):
                try:
                    ax6.axvline(dois_indexes[i], color='tab:red', linestyle='--')
                except IndexError:
                    pass
        elif alignment == 'age':
            for i in range(len(aoi)):
                ax6.axvline(aoi[i], color='tab:red', linestyle='--')

        # # Plot sound issues per session
        ax6.plot(x, sounds_mismatch, marker='o', ms=ms, lw=lw, color='tab:pink', label='Sounds mismatch')
        ax6.plot(x, message_count, marker='o', ms=ms, lw=lw, color='tab:purple', label='Message count')
        ax6.plot(x, no_sound, marker='o', ms=ms, lw=lw, color='tab:red', label='No sound')

        # ax6.set_xlabel('Days')
        ax6.set_xlim(xlim)
        ax6.set_xticklabels([])
        # ax6.xaxis.get_major_locator().set_params(integer=True)  # Force integers only in x ticks
        ax6.set_ylabel('Sound checks')
        ax6.legend(loc='upper right', fontsize='xx-small', frameon=True)
        ax6.spines['top'].set_visible(False)
        ax6.spines['bottom'].set_visible(False)
        # ax6.spines['right'].set_visible(False)

        # Instantiate a second axes that shares the same x-axis
        ax6_twin = ax6.twinx()
        ax6_twin.set_ylim([ax6.get_ylim()[0], ax6.get_ylim()[1]])  # Get ylims from ax6 and set them for ax6_twin
        ax6_twin.set_yticklabels([])
        ax6_twin.spines['top'].set_visible(False)
        ax6_twin.spines['bottom'].set_visible(False)

        time_end = time.time()
        runtime = time_end - time_start
        print("'Plot 6: 'sound checks' took", round(runtime, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 7: PROBABILITIES DIFFICULT TRIALS

        time_start = time.time()

        ax7 = plt.subplot2grid((9, 1), (7, 0), rowspan=1, colspan=1)

        # Plot vertical line for date of interest
        if alignment == 'n_sessions':
            for i in range(len(dois)):
                try:
                    ax7.axvline(dois_indexes[i], color='tab:red', linestyle='--')
                except IndexError:
                    pass
        elif alignment == 'age':
            for i in range(len(aoi)):
                ax7.axvline(aoi[i], color='tab:red', linestyle='--')

        # # Plot sound issues per session
        ax7.plot(x, p, marker='o', ms=ms, lw=lw, color='k')

        # ax7.set_xlabel('Days')
        ax7.set_xlim(xlim)
        ax7.set_xticklabels([])
        # ax7.xaxis.get_major_locator().set_params(integer=True)  # Force integers only in x ticks
        ax7.set_ylabel('P')
        # ax7.legend(loc='upper right', fontsize='xx-small', frameon=True)
        ax7.spines['top'].set_visible(False)
        ax7.spines['bottom'].set_visible(False)
        # ax7.spines['right'].set_visible(False)

        # Instantiate a second axes that shares the same x-axis
        ax7_twin = ax7.twinx()
        ax7_twin.set_yticklabels([])
        ax7_twin.spines['top'].set_visible(False)
        ax7_twin.spines['bottom'].set_visible(False)

        time_end = time.time()
        runtime = time_end - time_start
        print("'Plot 7: 'sound checks' took", round(runtime, 2), 'seconds to run')

        ################################################################################################################

        # PLOT 8: DELAY

        time_start = time.time()

        ax8 = plt.subplot2grid((9, 1), (8, 0), rowspan=1, colspan=1)

        # Plot vertical line for date of interest
        if alignment == 'n_sessions':
            for i in range(len(dois)):
                try:
                    ax8.axvline(dois_indexes[i], color='tab:red', linestyle='--')
                except IndexError:
                    pass
        elif alignment == 'age':
            for i in range(len(aoi)):
                ax8.axvline(aoi[i], color='tab:red', linestyle='--')

        # # Plot sound issues per session
        ax8.plot(x, var_delay, marker='o', ms=ms, lw=lw, color='k')

        ax8.set_xlabel(xlabel)
        ax8.set_xlim(xlim)
        # ax8.set_xticklabels([])
        # ax8.xaxis.get_major_locator().set_params(integer=True)  # Force integers only in x ticks
        ax8.set_ylabel('Delay')
        # ax8.legend(loc='upper right', fontsize='xx-small', frameon=True)
        ax8.spines['top'].set_visible(False)
        # ax8.spines['bottom'].set_visible(False)
        # ax8.spines['right'].set_visible(False)

        # Instantiate a second axes that shares the same x-axis
        ax8_twin = ax8.twinx()
        ax8_twin.set_yticklabels([])
        ax8_twin.spines['top'].set_visible(False)
        ax8_twin.spines['bottom'].set_visible(False)

        time_end = time.time()
        runtime = time_end - time_start
        print("'Plot 8: 'delays' took", round(runtime, 2), 'seconds to run')

        ################################################################################################################

        # Plot text
        ax.text(xlim[0], ax.get_ylim()[1], sum_text)

        ################################################################################################################
        # LEGACY PLOTS
        ################################################################################################################

        # # PLOT X: RESPONSES
        #
        # time_start = time.time()
        #
        # ax = plt.subplot2grid((8, 1), (0, 0), rowspan=1, colspan=1)
        #
        # # Plot horizontal lines
        # ax.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
        # ax.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
        # ax.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75
        #
        # # Plot vertical line for date of interest
        # if alignment == 'n_sessions':
        #     for i in range(len(dois)):
        #         try:
        #             ax9.axvline(dois_indexes[i], color='tab:red', linestyle='--')
        #         except IndexError:
        #             pass
        # elif alignment == 'age':
        #     for i in range(len(aoi)):
        #         ax9.axvline(aoi[i], color='tab:red', linestyle='--')
        #
        # # Plot response rate per session
        # ax.plot(x, response_rate, marker='o', ms=ms, lw=lw, color='black', label='Total')
        # ax.plot(x, response_rate_left, marker='o', ms=ms, lw=lw, color='tab:blue', label='Left')
        # ax.plot(x, response_rate_right, marker='o', ms=ms, lw=lw, color='tab:orange', label='Right')
        #
        # ax.set_xlim(xlim)
        # ax.set_xticklabels([])
        # ax.set_ylabel('Response\n(%)')
        # ax.set_ylim([0, 1.1])
        # ax.set_yticks(list(np.arange(0, 1.1, 0.1)))
        # ax.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        # # ax.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        # ax.legend(loc='lower right', fontsize='xx-small', frameon=True)
        # ax.spines['top'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # # ax.spines['right'].set_visible(False)
        #
        # # Instantiate a second axes that shares the same x-axis
        # ax_twin = ax.twinx()
        # ax_twin.set_ylim([0, 1.1])
        # ax_twin.set_yticks(list(np.arange(0, 1.1, 0.1)))
        # ax_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        # ax_twin.spines['top'].set_visible(False)
        # ax_twin.spines['bottom'].set_visible(False)
        #
        # time_end = time.time()
        # runtime = time_end - time_start
        # print("'Plot 0: 'responses' took", round(runtime, 2), 'seconds to run')

        ################################################################################################################

        # # PLOT 1: WATER
        #
        # time_start = time.time()
        #
        # ax1 = plt.subplot2grid((8, 1), (1, 0), rowspan=1, colspan=1)
        #
        # # Plot vertical line for date of interest
        # if alignment == 'n_sessions':
        #     for i in range(len(dois)):
        #         try:
        #             ax1.axvline(dois_indexes[i], color='tab:red', linestyle='--')
        #         except IndexError:
        #             pass
        # elif alignment == 'age':
        #     for i in range(len(aoi)):
        #         ax1.axvline(aoi[i], color='tab:red', linestyle='--')
        #
        # # Plot water per session
        # ax1.plot(x, water, marker='o', ms=ms, lw=lw, color='black', label='Total')
        # ax1.plot(x, water_left, marker='o', ms=ms, lw=lw, color='tab:blue', label='Left')
        # ax1.plot(x, water_right, marker='o', ms=ms, lw=lw, color='tab:orange', label='Right')
        #
        # ax1.set_xlim(xlim)
        # ax1.set_xticklabels([])
        # ax1.set_ylabel('Water')
        # ax1.set_ylim([0, water.max() + 100])
        # ax1.set_yticks(list(np.arange(0, water.max() + 100, 100)[0::5]))
        # ax1.legend(loc='lower right', fontsize='xx-small', frameon=True)
        # ax1.spines['top'].set_visible(False)
        # ax1.spines['bottom'].set_visible(False)
        # # ax1.spines['right'].set_visible(False)
        #
        # # Instantiate a second axes that shares the same x-axis
        # ax1_twin = ax1.twinx()
        # ax1_twin.set_ylim([0, water.max() + 100])
        # ax1_twin.set_yticks(list(np.arange(0, water.max() + 100, 100)[0::5]))
        # ax1_twin.set_yticklabels([])
        # ax1_twin.spines['top'].set_visible(False)
        # ax1_twin.spines['bottom'].set_visible(False)
        #
        # time_end = time.time()
        # runtime = time_end - time_start
        # print("'Plot 1: 'water' took", round(runtime, 2), 'seconds to run')

        ################################################################################################################

        # # PLOT 5: STAGES/SUBSTAGES/MOTOR
        #
        # time_start = time.time()
        #
        # ax5 = plt.subplot2grid((8, 1), (5, 0), rowspan=1, colspan=1)
        #
        # # Plot horizontal lines
        # # ax5.axhline(3, color='tab:gray', linestyle=':')  # Chance level
        # # ax5.axhline(6, color='tab:gray', linestyle=':')  # Accuracy 0.25
        # # ax5.axhline(9, color='tab:gray', linestyle=':')  # Accuracy 0.75
        #
        # # Plot vertical line for date of interest
        # if alignment == 'n_sessions':
        #     for i in range(len(dois)):
        #         try:
        #             ax5.axvline(dois_indexes[i], color='tab:red', linestyle='--')
        #         except IndexError:
        #             pass
        # elif alignment == 'age':
        #     for i in range(len(aoi)):
        #         ax5.axvline(aoi[i], color='tab:red', linestyle='--')
        #
        # # Plot stage/substage/motor per session
        # ax5.plot(x, stage, marker='o', ms=ms, lw=lw, color='black', label='Stage')
        # # ax5.plot(x, substage, marker='o', ms=ms, lw=lw, color='black', label='Substage')
        # ax5_twin = ax5.twinx()  # Instantiate a second axes that shares the same x-axis
        # ax5_twin.plot(x, motor, marker='o', ms=ms, lw=lw, color='tab:gray', label='Motor')
        #
        # # ax5.set_xlabel('Days')
        # ax5.set_xlim(xlim)
        # ax5.set_xticklabels([])
        # ax5.set_ylim()
        # ax5.set_ylabel('Stage')
        # ax5.set_yticks(stage.unique())
        # # ax5.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        # # ax5.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        # ax5.legend(loc='lower right', fontsize='xx-small', frameon=True)
        # ax5.spines['top'].set_visible(False)
        # ax5.spines['bottom'].set_visible(False)
        # # ax5.spines['right'].set_visible(False)
        #
        # # Instantiate a second axes that shares the same x-axis
        # # ax5_twin = ax5.twinx()
        # # ax5_twin.set_ylim([0, 4])
        # ax5_twin.set_yticks(motor.unique())
        # # ax5_twin.set_yticklabels(['', '', '', '', '', '', '', '', '', '', ''])
        # ax5_twin.set_ylabel('Motor')
        # ax5_twin.spines['top'].set_visible(False)
        # ax5_twin.spines['bottom'].set_visible(False)
        #
        # # Make shared legend for both axis
        # lines_1, labels_1 = ax5.get_legend_handles_labels()
        # lines_2, labels_2 = ax5_twin.get_legend_handles_labels()
        # lines = lines_1 + lines_2
        # labels = labels_1 + labels_2
        # ax5.legend(lines, labels, loc='lower right', fontsize='xx-small', frameon=True)
        #
        # time_end = time.time()
        # runtime = time_end - time_start
        # print("'Plot 5: 'stages/motor' took", round(runtime, 2), 'seconds to run')

        ################################################################################################################
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
        columns = ['Dates', 'DoW', 'Age', 'Subject', 'Board', 'Trials', 'TrialsLeft', 'TrialsRight', 'ChoseLeft',
                   'ChoseRight',
                   'Hits', 'HitsLeft', 'HitsRight', 'HitsRep', 'HitsAlt', 'Errors', 'ErrorsLeft', 'ErrorsRight',
                   'Performance', 'PerformanceLeft', 'PerformanceRight', 'Responses', 'ResponsesLeft', 'ResponsesRight',
                   'Repetitions', 'RepsLeft', 'RepsRight', 'RepRateLeft', 'RepRateRight', 'Alternations', 'AltsLeft',
                   'AltsRight', 'AltRateLeft', 'AltRateRight', 'Accuracy', 'AccuracyLeft', 'AccuracyRight', 'AccMaxEvi',
                   'LateralBias', 'AccuracyRep', 'AccuracyAlt', 'RepBias', 'CorrRepBias', 'CorrAltBias', 'Misses',
                   'MissesLeft', 'MissesRight', 'MissRate', 'MissRateLeft', 'MissRateRight', 'Rewards', 'RewardsLeft',
                   'RewardsRight', 'Water', 'WaterLeft', 'WaterRight', 'Stage', 'SoundsMismatch', 'NoSound',
                   'MessageCount', 'P', 'VarDelay', 'PCRight', 'xPCRight', 'yPCRight', 'FitPCRight', 'FitErrorPCRight',
                   'ParamsPCRight', 'SensitivityPCRight', 'BiasPCRight', 'LapseRight', 'LapseLeft', 'PCRep', 'xPCRep',
                   'yPCRep', 'FitPCRep', 'FitErrorPCRep', 'ParamsPCRep', 'SensitivityPCRep', 'BiasPCRep', 'LapseRep',
                   'LapseAlt', 'Drug']

        data = list(
            zip(dates, dow, age, subject, board, trials, trials_left, trials_right, chose_left, chose_right, hits,
                hits_left, hits_right, hits_rep, hits_alt, errors, errors_left, errors_right, performance,
                performance_left, performance_right, responses, responses_left, responses_right, repetitions,
                reps_left, reps_right, rep_rate_left, rep_rate_right, alternations, alts_left, alts_right,
                alt_rate_left, alt_rate_right, accuracy, accuracy_left, accuracy_right, accuracy_max_evi,
                lateral_bias, accuracy_rep, accuracy_alt, rep_bias, corr_rep_bias, corr_alt_bias, misses,
                misses_left, misses_right, miss_rate, miss_rate_left, miss_rate_right, rewards, rewards_left,
                rewards_right, water, water_left, water_right, stage, sounds_mismatch, no_sound, message_count, p,
                var_delay, pc_right, xdata_pc_right, ydata_pc_right, fit_pc_right, fit_error_pc_right, params_pc_right,
                sensitivity_pc_right, bias_pc_right, lapse_right, lapse_left, pc_rep, xdata_pc_rep, ydata_pc_rep,
                fit_pc_rep, fit_error_pc_rep, params_pc_rep, sensitivity_pc_rep, bias_pc_rep, lapse_rep, lapse_alt, drug))

        df_intersession = pd.DataFrame(data=data, columns=columns)

    # Select the output folder for the .csv file and create it if it doesn't exist
    folder_csv_out = Path.home() / 'PycharmProjects' / 'intersession' / experiment
    if not os.path.exists(folder_csv_out):
        # os.mkdir(folder_csv_out)
        folder_csv_out.mkdir(parents=True, exist_ok=True)

    if to_csv:
        df_intersession.to_csv(Path(folder_csv_out / (setup + '_intersession.csv')),
                               index=False)  # index=False to avoid the 'Unmmaed: 0' column

    # This block needs to be the last otherwise it sends the file too soon and corrupted
    if send_slack:
        # with open('/home/alexis/slack_bot_token', 'r') as f:  # Get slack bot token Alexis personal laptop
        with open(Path.home() / 'slack_bot_token', 'r') as f:  # Get slack bot token
            slack_bot_token = f.read().replace('\n', '')

        os.environ['SLACK_BOT_TOKEN'] = slack_bot_token
        # filepath = folder_pdf_out + '/' + df.Session.unique()[0]
        filepath = Path(folder_pdf_out / filename)
        filepath = str(filepath)  # filepath, input to slack api method files.upload, used by function slack_spam,
        # requires the file path as a str
        slack_spam(msg='Hey buddy!', filepath=filepath, userid='#pv_nmdar_eranet_reports')  # Alexis: 'U01DDHH7LLX'

    return df_intersession


@timer
def do_intersessions(experiment='2AFC_6', alignment='n_sessions', to_csv=True,
                     send_slack=False):
    """Do the intersessions for all animals of a given batch (experiment)"""

    # Update glued sessions first
    update_glued_sessions(experiment=experiment)

    if experiment is None:

        folder = Path.home() / 'pv_nmdar_eranet' / 'experiments'

        experiments = os.listdir(folder)  # List experiments
        experiments.sort()  # Sort them by name

        experiments_to_remove = ['.idea', 'Daily check', 'WaterCalibration']
        for _ in range(len(experiments_to_remove)):
            try:
                experiments.remove(experiments_to_remove[_])
            except ValueError:
                pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    # print('Doing intersession reports of: ' + animal)
    folder = Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment
    animals = os.listdir(folder)
    animals = [animals for animals in animals if animals.endswith('.csv') and len(animals) == 7]
    animals.sort()

    username = os.getlogin()

    for i in range(len(animals)):
        path = Path(folder / animals[i])
        df = pd.read_csv(path)
        board = df.Board.unique()[0]  # Bpod (box)
        box = int(board[-1])
        flag = False  # flag always needs to be initialized as False

        # Only do the intersession reports of the animals training in that PC (so the data is stored locally)
        if username == 'setup0':
            if box == 0:
                flag = True
        elif username == 'setup1':
            if 0 < box < 5:
                flag = True
        elif username == 'setup2':
            if box > 4:
                flag = True
        elif username != 'setup0' or username != 'setup1' or username != 'setup2':
            flag = True

        if df.Protocol.unique()[0] != 'stage_training_v6':  # In Ephys PC (setup0) there is only one experiment (Ephys)
            # instead of the 2AFC_X nomenclature followed in the other PCs
            flag = False

        if flag:
            try:
                intersession_within_animal(path, alignment=alignment, to_csv=to_csv, send_slack=send_slack)
            except:
                print(f'Could not do intersession report of animal {i}')


@timer
def glue_animals_intersessions(experiment='2AFC_6', filter_drug=False, update=True, to_csv=True):
    """
    Concatenate all intersession .csv files from each animal into a single .csv file
    :param update: If True update first the glued sessions
    :param to_csv: True for saving the output DataFrame, default is False (do not save)
    :return: DataFrame with all the intersession concatenated
    """

    # Update first the glued sessions
    if update:
        update_glued_sessions(experiment=experiment)  # Update glued sessions first

    # Get the path to the data
    experiment, path_experiment = get_experiment(experiment=experiment, path_session='intersession')

    intersessions = os.listdir(path_experiment)  # Get list of
    intersessions.sort()
    intersessions = [x for x in intersessions if x.endswith('.csv')]  # Get rid of non csv files
    # print(intersessions)

    df = pd.DataFrame()

    for i in range(len(intersessions)):
        df_intersession = pd.read_csv(path_experiment / intersessions[i])
        if filter_drug:
            df_intersession = filter_drug_sessions(df_intersession)
        df = pd.concat([df, df_intersession])

    if to_csv:
        df.to_csv(Path(path_experiment / (experiment + '_intersessions' + '.csv')), index=False)
        # index=False to avoid the 'Unnamed: 0' column

    return df


@timer
def intersession_across_boxes(experiment='2AFC_6', variable='Accuracy', mean=False,
                              update=False, to_csv=False):
    """
    Plot intersession across boxes (Board). X axis is still Dates, but in each plot there's one trace per animal.
    There's one plot per box
    :param protocol:
    :param experiment:
    :param update:
    :param to_csv:
    :return:
    """

    # Dates of interest (format: 'yyyy-mm-dd') to plot a vertical line
    # For 2AFC_6 (batch 6)
    doi_24 = '2025-06-26'  # First attempt ti fix XYZ coordinates per mouse
    doi_25 = '2025-07-15'  # Drug experiment beginning
    dois = [doi_24, doi_25]  # Dates of interest

    df = glue_animals_intersessions(experiment=experiment, update=update, to_csv=to_csv)
    df['Dates'] = pd.to_datetime(df['Dates'])
    df = df[df.Subject != 11]  # Drop animal 011 from analysis
    boxes = list(df.Board.unique()) # Get the boxes
    boxes.sort()  # Sort boxes by number
    animals_per_box = {box: df[df.Board == box].Subject.unique() for box in boxes}  # Get the animals per box

    if variable == 'Accuracy':
        ylim = [0.4, 1]  # Y limits for accuracy
        hline_y = 0.5  # Chance level for accuracy
        y_ticks = [0.5, 0.75, 1]  # Y ticks for accuracy
        y_ticklabels = ['0.5', '0.75', '1']  # Y tick labels for accuracy
    elif variable == 'LateralBias':
        ylim = [-1, 1]
        hline_y = 0
        y_ticks = [-1, -0.5, 0, 0.5, 1]
        y_ticklabels = ['L', '-0.5', '0', '0.5', 'R']

    # Make a plot with a subplot per box
    figsize = (11.69, 8.27)  # A4 size in inches
    plt.figure(figsize=figsize)

    for i, box in enumerate(boxes):
        ax = plt.subplot(len(boxes), 1, i + 1)  # Create a subplot for each box
        df_box = df[df.Board == box]  # Filter the DataFrame for the current box

        # Collect all unique dates (across animals in this box)
        unique_dates = sorted(df_box.Dates.unique())
        session_indexes = np.arange(len(unique_dates))

        # Subsample ticks every 5 days
        tick_step = 5
        tick_dates = unique_dates[::tick_step]
        tick_labels = session_indexes[::tick_step]

        # Find the indexes of the dates of interest and plot vertical lines
        for doi in dois:
            doi_dt = pd.to_datetime(doi)
            if doi_dt in unique_dates:
                ax.axvline(doi_dt, color='tab:red', linestyle='--')
            else:
                print(f'Date of interest {doi} not found in box {box}')

        # Plot
        ax.axhline(hline_y, color='tab:gray', linestyle='--')  # Chance level
        if mean:
            # Plot mean with std as errobars per box in black
            df_mean = df_box.groupby('Dates')[variable].mean()
            ax.plot(df_mean.index, df_mean, color='k', label='Mean')
            df_std = df_box.groupby('Dates')[variable].std()
            ax.fill_between(df_mean.index, df_mean - df_std, df_mean + df_std, color='k', alpha=0.2, label='Std')
        else:
            # Plot each animal in the box
            for animal in animals_per_box[box]:
                df_animal = df_box[df_box.Subject == animal].sort_values('Dates')
                ax.plot(df_animal.Dates, df_animal[variable], marker='o', label=animal)

        ax.set_ylim(ylim)
        ax.set_yticks(y_ticks,y_ticklabels)
        ax.set_ylabel(f'{box}')
        ax.legend(loc='upper left', frameon=False)

        # Replace date ticks with session numbers
        ax.set_xticks(tick_dates)
        ax.set_xticklabels(tick_labels)

        if i == 0:
            ax.set_title(variable)

        if i != 3:  # Remove lower axes line unless last plot (Box 4)
            sns.despine(bottom=True, ax=ax)
            ax.set_xticklabels([])  # Remove x-tick labels for all but the last plot
        else:
            sns.despine(ax=ax)
            ax.set_xlabel('N sessions')

@timer
def learning_curves(experiment='2AFC_4', alignment='n_sessions', offset=None, save=False):
    """
    Plot the learning trajectories (accuracy vs time) of all animals of a given batch
    """

    time_start = time.time()

    # Get the path to the data
    experiment, folder_in = get_experiment(experiment, session='intersession')
    # folder_in = Path.home() / 'PycharmProjects' / 'intersession' / experiment

    animals = os.listdir(folder_in)
    animals = [animals for animals in animals if animals.endswith('.csv')]
    animals.sort()

    # Alignment (x-axis)
    if alignment == 'n_sessions':  # Number of training sessions. Starts at 0
        x = 'SessionNumber'
        xlabel = 'N sessions'
    elif alignment == 'age':  # Age of the animal in days. Doesn't start at 0
        x = 'Age'
        xlabel = 'Age (days)'
    filename = f'_learning_trajectories_({x}_aligned)'

    df_all_intersessions = pd.DataFrame()  # Create empty DataFrame
    first_p_session_indexes = []
    animals_discarded = 0
    plt.figure()

    for _ in range(len(animals)):
        path = folder_in / animals[_]
        df = pd.read_csv(path)
        # df = df[df.P == 0]  # Only sessions without evidences
        # df = df[df.P > 0]  # Only sessions with evidences

        # Get the first session in which the evidences were introduced
        try:
            first_p_session_index = np.where(df.P > 0)[0][0]
        except IndexError:  # index 0 is out of bounds for axis 0 with size 0 (no sessions with p > 0)
            animals_discarded += 1
            continue  # Skip iteration if animal didn't make it until evidences
            # first_p_session_index = np.nan
        first_p_session_indexes.append(first_p_session_index)

        # Offset
        if offset == 'first_p':
            df = df[first_p_session_indexes[-1]:].reset_index()

        session_number = df.index.to_list()  # Get session numbers
        df.insert(1, 'SessionNumber', session_number)  # Insert session number in column 1 of df
        print(f'Subject {str(df.Subject.unique())[1:-1]}: {len(df)} sessions')
        df_all_intersessions = pd.concat([df_all_intersessions, df])
        # plt.plot(df[x], df.Accuracy, color='tab:gray', alpha=0.5)  # Plot individual animals

    df_all_intersessions.VarDelay.fillna(0, inplace=True)  # Set the nan of delay to 0

    # Hits
    hits = df_all_intersessions.groupby(x).Hits.sum().astype('int')
    hits_left = df_all_intersessions.groupby(x).HitsLeft.sum().astype('int')
    hits_right = df_all_intersessions.groupby(x).HitsRight.sum().astype('int')

    # Responses (valid trials)
    responses = df_all_intersessions.groupby(x).Responses.sum()
    responses_left = df_all_intersessions.groupby(x).ResponsesLeft.sum()
    responses_right = df_all_intersessions.groupby(x).ResponsesRight.sum()

    # Accuracy (hit rate)
    accuracy = hits / responses
    accuracy_left = hits_left / responses_left
    accuracy_right = hits_right / responses_right

    # Plot mean accuracy
    label = f'({experiment}, N={len(animals) - animals_discarded}/{len(animals)})'
    # plt.plot(accuracy_left, color='tab:blue', linewidth=3, alpha=0.25)
    # plt.plot(accuracy_right, color='tab:orange', linewidth=3, alpha=0.25)
    # plt.plot(df_all_intersessions.groupby(x).LateralBias.mean() + 0.5, color='tab:red', linestyle='--', linewidth=3, alpha=0.25)
    # plt.plot(accuracy, color='k', linewidth=3)
    plt.plot(df_all_intersessions.groupby(x).Accuracy.mean(), color='k', linewidth=3,
             label='Acc')  # Almost the same, but not
    plt.plot(df_all_intersessions.groupby(x).AccMaxEvi.mean(), color='tab:green', linewidth=3, alpha=0.75,
             label='Acc. Max. Evi.')  # Mean accuracy at max evidence
    # plt.plot(df_all_intersessions.groupby(x).P.mean() + 0.5, color='r', linewidth=3, alpha=0.75, label='P (+0.5)')  # Mean P (difficulty)
    plt.plot(df_all_intersessions.groupby(x).P.mean(), color='tab:red', linewidth=3, alpha=0.75,
             label='P')  # Mean P (difficulty)
    # plt.plot(df_all_intersessions.groupby(x).VarDelay.mean(), color='tab:pink', linewidth=3, alpha=0.75, label='Delay')  # Mean P (difficulty)

    # plt.axvline(np.mean(first_p_session_indexes), color='r')
    # label = f'({experiment}, N={len(animals) - animals_discarded}/{len(animals)})'
    plt.title(f'Learning trajectories ({experiment}, N={len(animals) - animals_discarded}/{len(animals)})')

    if offset == 'first_p':
        xlabel = xlabel + ', ' + offset + ' ' + 'offset'
        filename = filename + '_' + offset + '_offset'

    plt.xlabel(xlabel)
    plt.ylim([-0.05, 1])
    # plt.ylim([0.4, 1])
    plt.ylabel('Accuracy')

    if save:
        folder_out = Path.home() / 'Documentos' / 'learning curves' / experiment
        save_fig(folder_out, filename)
        plt.close()

    time_end = time.time()
    runtime = time_end - time_start
    print('The script took', round(runtime, 2), 'seconds to run')


########################################################################################################################

# For debugging
# path = Path.home() / 'PycharmProjects' / 'glue_sessions' / '2AFC_4' / '911.csv'  # Where the data for all animals is
# alignment = 'age'
# to_csv = False
# send_slack = False
# intersession_within_animal(path, alignment='n_sessions', to_csv=False, send_slack=False)
# do_intersessions(protocol='stage_training_v4', experiment='2AFC_4', alignment='age', to_csv=True, send_slack=False)

# learning_curves(experiment='2AFC_2', alignment='n_sessions', offset=None, save=True)
# learning_curves(experiment='2AFC_2', alignment='n_sessions', offset='first_p', save=True)
# learning_curves(experiment='2AFC_2', alignment='age', offset=None, save=True)
# learning_curves(experiment='2AFC_2', alignment='age', offset='first_p', save=True)
# learning_curves(experiment='2AFC_3', alignment='n_sessions', offset=None, save=True)
# learning_curves(experiment='2AFC_3', alignment='n_sessions', offset='first_p', save=True)
# learning_curves(experiment='2AFC_3', alignment='age', offset=None, save=True)
# learning_curves(experiment='2AFC_3', alignment='age', offset='first_p', save=True)
# learning_curves(experiment='2AFC_4', alignment='n_sessions', offset=None, save=True)
# learning_curves(experiment='2AFC_4', alignment='n_sessions', offset='first_p', save=True)
# learning_curves(experiment='2AFC_4', alignment='age', offset=None, save=True)
# learning_curves(experiment='2AFC_4', alignment='age', offset='first_p', save=True)

# M = smf.mixedlm("AccMaxEvi ~ SessionNumber + Age + DoW + P", data=df_all_intersessions, groups=df_all_intersessions.Subject).fit()
# M.summary()
#
# # Delay only in batch 2AFC_4
# M = smf.mixedlm("AccMaxEvi ~ SessionNumber + Age + DoW + P + VarDelay", data=df_all_intersessions, groups=df_all_intersessions.Subject).fit()
# M.summary()
