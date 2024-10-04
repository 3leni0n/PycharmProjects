# IMPORTS
import numpy as np
import pandas as pd
import utils
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

##################### QUALITY REPORT FUNCTONS #####################

def acc_error_time(df, event, axes):
    if event=='corridor_diff':
        x_var, label = 'STATE_Fixation1_START_align', 'CORRIDOR'
    elif event == 'delay_diff':
        x_var, label = 'delay_start', 'DELAY'
    elif event=='response_diff':
        x_var, label = 'STATE_Response_window_END_align', 'RESPONSE'
    else:
        ('WARNING EVENT INCORRECTLY SPECIFIED!')

    sns.scatterplot(x=x_var, y=event, hue='trial', data=df, palette='Spectral', linewidths=1, s=20, ax=axes)
    init_value=df[event].iloc[df[event].first_valid_index()]
    end_value= df[event].iloc[df[event].last_valid_index()]
    # Draw a horizontal line
    axes.axhline(y=init_value, linestyle=':', linewidth=1, color='silver')
    # Axis parameters
    axes.set_title('CUMULATIVE DIFFERENCE ' + str(label) + ': ' + str(round(end_value-init_value, 3)*1000) + ' ms', fontsize=8)
    axes.set_ylabel('Time difference ttl behavior (s)')
    axes.set_xlabel('Trials')
    axes.get_legend().remove()

def corr_resp_diff(df, axes):
    df['corr_resp_state_diff'] = df['STATE_Response_window_END_align'] - df['STATE_Fixation1_START_align']
    df['corr_resp_ttl_diff'] =   df['response_ttl_fix'] - df['corridor_ttl_fix']
    median = (df['corr_resp_ttl_diff'] - df['corr_resp_state_diff']).median() * 1000

    # plot
    sns.scatterplot(x='trial', y='corr_resp_state_diff', data=df, color='firebrick', linewidths=0, s=100,
                    ax=axes, label='Behavior')
    sns.scatterplot(x='trial', y='corr_resp_ttl_diff', data=df, color='cornflowerblue', linewidths=0, s=10,
                    ax=axes, label='TTLs')
    # Axis parameters
    axes.set_title('MEDIAN DIFFERENCE ' + str(round(median, 3)) + ' ms', fontsize=8)
    axes.set_ylabel('Time difference corridor - response  (s)')
    axes.set_xlabel('Trials')


def alignment_check(df, event, x_min, x_max, axes):
    if event== 'corridor_ttl_fix':
        color = 'purple'
        event2 = 'STATE_Fixation1_START_align'
    elif event== 'delay_ttl_fix':
        color = 'yellowgreen'
        event2 = 'delay_start'
    elif event== 'response_ttl_fix':
        color = 'salmon'
        event2 = 'STATE_Response_window_END_align'

    for timestamp in df[event2].values:
        axes.axvline(x=timestamp, color=color, linestyle='--', alpha=0.8)
    sns.scatterplot(x=df[event], y=df['subject'], linewidths=1, s=1000, marker='|',
                    color='black')
    axes.set_ylabel(event)
    axes.set_yticklabels([''])
    axes.set_xlabel('Session time (sec)')
    axes.set_xlim(x_min, x_max)
    sns.despine()


def quality_report(df_behavior, df_spikes, ttl_min_clock, ttl_max_clock,  spike_min_clock, spike_max_clock, alignment_diff_c,
                   missing_corridor_ttl, missing_response_ttl, missing_delay_ttl, save_path):

    print('Doing Quality Report...')

    ########## BEHAVIOR TIMES  ##########
    beh_starting_datetime = datetime.fromtimestamp(df_behavior['STATE_Start_task_START'].iloc[0])
    beh_starting_datetime -= timedelta(hours=1)  # Subtract one hour
    beh_starting_dt = beh_starting_datetime.strftime('%Y-%m-%d %H:%M:%S')

    try:
        beh_finishing_datetime = datetime.fromtimestamp(df_behavior['STATE_Exit_END'].iloc[-1])
    except:
        beh_finishing_datetime = datetime.fromtimestamp(df_behavior['STATE_Exit_END'].iloc[-2])

    beh_finishing_datetime = beh_finishing_datetime - timedelta(hours=1)  # Subtract one hour
    beh_finishing_dt = beh_finishing_datetime.strftime('%Y-%m-%d %H:%M:%S')

    beh_dur_timestamp = beh_finishing_datetime - beh_starting_datetime
    beh_dur_max_clock = beh_dur_timestamp.total_seconds()

    n_trials =  df_behavior.trial.max()
    n_memory_trials= df_behavior.loc[df_behavior['trial_type'].str.contains('WM')]['trial_type'].count()


    ########## UNITS CLASSIFICATION ############
    try:
        good_units = df_spikes.groupby('group')['cluster_id'].unique()['good']
    except:
        good_units =np.array([])
    multi_units =df_spikes.groupby('group')['cluster_id'].unique()['mua']


    ########## LATENCIES BETWEEN BEHAVIOR AND EPHYS  ##########
    df_behavior['corridor_diff'] = df_behavior['corridor_ttl_fix'] - df_behavior['STATE_Fixation1_START_align']
    df_behavior['response_diff'] = df_behavior['response_ttl_fix'] - df_behavior['STATE_Response_window_END_align']
    df_behavior['delay_diff'] = df_behavior['delay_ttl_fix'] - df_behavior['delay_start']
    median_time_diff = df_behavior['corridor_diff'].median() #median times


    ######## WARNINGS #########
    warning_list = []

    # Behavioral session should start after the recording onset
    if alignment_diff_c <0:
        warning_list.append(' WARNING! Behavior starting  beofre the recording ')

    # Behavioral session should finish before the recording
    if ttl_max_clock < beh_dur_max_clock:
        warning_list.append(' WARNING! Behavior finishing after the recording ')

    # TTL clock should start before the sorted spikes or same time
    if spike_min_clock <= ttl_min_clock:
        diff= abs(ttl_min_clock-spike_min_clock)
        warning_list.append(' WARNING! TTL starting after the spikes: '+ str(round(diff, 3)) +'s ')

    # TTL clock should finish after the sorted spikes or same time
    if ttl_max_clock <= spike_max_clock:
        diff = abs(spike_max_clock - ttl_max_clock)
        warning_list.append(' WARNING! TTL finishing after the spikes,   Diff: ' +str(round(diff, 3)) +'s ')

    # Accumulative error of ttls over the session
    if round(df_behavior['corridor_diff'].iloc[-1], 3) != round(df_behavior['corridor_diff'].iloc[0], 3):
        diff = abs(df_behavior['corridor_diff'].iloc[-1]-df_behavior['corridor_diff'].iloc[0])
        warning_list.append(' WARNING! Accumulated error over time '+str(round(diff, 3)) +'s ')

    # More TTLs than expected
    for i in ['corridor_ttl', 'delay_ttl', 'response_ttl']:
        more_than_one = df_behavior[i].apply(lambda x: len(x) > 1)
        if more_than_one.any():
            problem_trials= df_behavior.loc[more_than_one, 'trial'].values
            warning_list.append('WARNING! More TTLS than expected ' + str(i)+ ' in trials: '+str(problem_trials))

    # Missing TTLs
    for i in [missing_corridor_ttl, missing_response_ttl]: #missing_delay_ttl
        if len(i) != 0:
            warning_list.append('WARNING! Missing TTLS in '+  str(len(i))+ ' trials')

    # Trials with missaligned ttls
    for i in ['corridor_diff', 'delay_diff', 'response_diff']:
        problem_trials=df_behavior.loc[(df_behavior[i] < -0.1) | (df_behavior[i] > 0.1), 'trial'].values
        if len(problem_trials)!=0:
            warning_list.append('WARNING! Trials with TTLs missaligned ' + str(i) + ' in trials: ' + str(problem_trials))

    # # Median time diff should be similar to first time diff
    # if abs(median_time_diff - df_behavior['corridor_diff'].iloc[0]) > 0.1:
    #     diff = abs(median_time_diff - df_behavior['corridor_diff'].iloc[0])
    #     warning_list.append(' WARNING! Median time for alignment not properly calculated ' +str(round(diff, 3)) +'s ')

    # Sesions usually contain few good units
    if len(good_units) == 0:
        warning_list.append(' WARNING! No good units' )

    # Group every two elements and join with a newline character
    warnings_text = '\n'.join([' '.join(warning_list[i:i + 2]) for i in range(0, len(warning_list), 2)])

    ######### PLOT ##########
    with PdfPages(save_path) as pdf:
        plt.figure(figsize=(11.7, 15))


        ########## HEADING ########
        s1 = ('BEHAVIOR       -->  DURATION ' + str(round(beh_dur_max_clock / 60, 3)) + ' min' +
                        '  /  Nº TOTAL TRIALS: ' + str(n_trials) +
                        '  /  Nº MEMORY TRIALS: ' + str(n_memory_trials) +
                        '  /  STARTING TIME: ' + str(beh_starting_dt) +
                        '  /  FINISHING TIME: ' + str(beh_finishing_dt) + '\n')

        s2 = ('EPHYS TTLs      -->  DURATION ' + str(round(ttl_max_clock / 60, 3)) + ' min'  +
              '  /  Nº CORR STARTS: ' + str(df_behavior['corridor_ttl_fix'].nunique()) +
              '  /  Nº MEMORY TRIALS: ' + str(df_behavior['delay_ttl_fix'].nunique()) +
              '  /  Nº RESP WIN FINISHED: ' + str(df_behavior['response_ttl_fix'].nunique()) + '\n')

        s3 = ('EPHYS SPIKETIMES-->  DURATION ' + str(round(spike_max_clock / 60, 3)) + ' min' +
              '  / Nº GOOD UNITS: ' + str(len(good_units)) +
              '  / Nº MULTI UNITS: ' + str(len(multi_units)) + '\n')

        s4 = ('EPHYS - BEHAVIOR  ALIGMENT EVENT LATENCY --> FIRST: ' + str(round(df_behavior['corridor_diff'].iloc[0] , 3)) + 's, ' +
                                                            'LAST:' + str(round(df_behavior['corridor_diff'].iloc[-1], 3)) + 's, ' +
                                                            'MEDIAN: ' + str(round(median_time_diff, 3)) + 's, ' + '\n')

        s5 = ('\n'+ '---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------' +'\n')

        s6 = (warnings_text+ '\n'+
              '---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------' +'\n')


        ########### PLOT ############

        # HEADER
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=10, colspan=22)
        axes.text(0.1, 0.9, s1 + s2 + s3 + s4 + s5 + s6, fontsize=8, transform=plt.gcf().transFigure)  # HEADER


        # PLOT 1: CORRIDOR ACCUMULATED ERROR OVER TIME
        # recycled axes form header
        acc_error_time(df_behavior, 'corridor_diff',axes)

        # PLOT 2: DELAY ACCUMULATED ERROR OVER TIME
        axes = plt.subplot2grid((50, 50), (13, 0), rowspan=10, colspan=22)
        acc_error_time(df_behavior, 'delay_diff', axes)

        # PLOT 3: RESPONSE ACCUMULATED ERROR OVER TIME
        axes = plt.subplot2grid((50, 50), (26, 0), rowspan=10, colspan=22)
        acc_error_time(df_behavior, 'response_diff',axes)

        # PLOT 4: ERROR WITHIN TRIAL
        axes = plt.subplot2grid((50, 50), (0, 26), rowspan=10, colspan=23)
        corr_resp_diff(df_behavior, axes)

        # PLOT 5: CORRIDOR ALIGMENT CHECK FIRST & LAST
        axes = plt.subplot2grid((50, 50), (13, 26), rowspan=5, colspan=23)
        max_timestamp= df_behavior.corridor_ttl_fix.max()
        x_min= 0
        x_max= 800
        alignment_check(df_behavior, 'corridor_ttl_fix', x_min, x_max, axes)
        axes = plt.subplot2grid((50, 50), (20, 26), rowspan=5, colspan=23)
        x_min = max_timestamp-800
        x_max = max_timestamp
        alignment_check(df_behavior, 'corridor_ttl_fix', x_min, x_max, axes)

        # PLOT 5: RESPONSE ALIGMENT CHECK FIRST & LAST
        axes = plt.subplot2grid((50, 50), (28, 26), rowspan=5, colspan=23)
        max_timestamp = df_behavior.corridor_ttl_fix.max()
        x_min = 0
        x_max = 800
        alignment_check(df_behavior, 'response_ttl_fix', x_min, x_max, axes)
        axes = plt.subplot2grid((50, 50), (35, 26), rowspan=5, colspan=23)
        x_min = max_timestamp - 800
        x_max = max_timestamp
        alignment_check(df_behavior, 'response_ttl_fix', x_min, x_max, axes)

        ########### SAVING AND CLOSING PAGE ###########
        sns.despine()
        pdf.savefig()
        plt.close()

        print('New quality report completed successfully!')





######################## EPHYS REPORT ########################

#### PLOT FUNCTIONS ####
def plot_raster(x_pos, y_pos, colspan, unit_df, event, hue, y_max_trial,  title_pos, stim_o, stim_c, stim_f, linesxl, linesxr, linesy):
    to_plot = unit_df.loc[(unit_df[event['align']] >= event['boundaries'][0]) &
                          (unit_df[event['align']] <= event['boundaries'][1])]
    axes = plt.subplot2grid((50, 60), (y_pos, x_pos), rowspan=10, colspan=colspan)
    sns.scatterplot(data=to_plot, x=event['align'], y='trial_index_response', ax=axes,
                    hue=hue, hue_order=stim_o, palette=stim_c, s=5, marker='o')
    sns.scatterplot(x=linesxl, y=linesy, ax=axes, color='black', s=5, marker='|')
    sns.scatterplot(x=linesxr, y=linesy, ax=axes, color='black', s=5, marker='|')
    axes.axvline(0, color='black', linewidth=1)
    axes.set_ylim(-1, y_max_trial)
    axes.set_xlim(event['boundaries'][0], event['boundaries'][1])
    axes.text(title_pos, 1.05, event['name'], ha='center', fontweight='bold', transform=axes.transAxes, fontsize=9)
    #axes
    if event['first']:
        axes.set_ylabel('Trial index')
        spines = ['top', 'right']
        lines = [Line2D([0], [0], color=c, marker='o', markerfacecolor=c) for c in stim_c]
        axes.legend(lines, stim_f, title='Choice')
    else:
        spines = ['top', 'right', 'left']
        axes.yaxis.set_visible(False)
        axes.get_legend().remove()
    for spine in spines:
        axes.spines[spine].set_visible(False)

    return axes


def plot_psth(x_pos, y_pos, colspan,  df_beh_conv, event, hue, y_max_fr, hue_order, hue_color, hue_label):
    to_plot = df_beh_conv.loc[(df_beh_conv[event['align']] >= event['boundaries'][0]) &
                              (df_beh_conv[event['align']] <= event['boundaries'][1])]
    to_plot = to_plot.groupby([event['align'], hue])['conv_firing'].agg(['mean', 'std']).reset_index()
    to_plot['error_lower'] = to_plot['mean'] - to_plot['std'] / 2
    to_plot['error_upper'] = to_plot['mean'] + to_plot['std'] / 2

    axes = plt.subplot2grid((50, 60), (y_pos, x_pos), rowspan=8, colspan=colspan)
    for choice, color, label in zip(hue_order, hue_color, hue_label):
        to_plot2 = to_plot[to_plot[hue] == choice]
        axes.plot(to_plot2[event['align']], to_plot2['mean'], color=color, label=label)
        axes.fill_between(to_plot2[event['align']], to_plot2['error_lower'], to_plot2['error_upper'],
                          color=color, alpha=.2)
    # axes
    axes.axvline(0, color='black', linewidth=1)
    axes.set_ylim(0, y_max_fr+5)
    axes.set_xlim(event['boundaries'][0], event['boundaries'][1])
    axes.set_xlabel('Time from onset (s)')
    if event['first']:
        spines = ['top', 'right']
        axes.set_ylabel('Spike rate (spikes/s)')
    else:
        spines = ['top', 'right', 'left']
        axes.yaxis.set_visible(False)
        if event['name']== 'response_reward':
            # lines = [Line2D([0], [0], color=c, marker='o', markerfacecolor=c) for c in hue_color]
            # axes.legend(lines, hue_label, title='Choice')
            axes.legend()
    for spine in spines:
        axes.spines[spine].set_visible(False)
    return axes



def plot_autocorr(x_pos, y_pos, bins_auto, autocorr_array):
    axes = plt.subplot2grid((50, 60), (y_pos, x_pos), rowspan=10, colspan=12)
    axes.bar(bins_auto, autocorr_array, width=1, color='gray', linewidth=0.3)
    axes.set_xlabel('') #Time (ms)
    axes.set_title('Autocorrelogram')
    axes.axvline(2, color='black', linewidth=0.2)
    axes.axvline(-2, color='black', linewidth=0.2)
    spines = ['top', 'right']     # remove axes spines
    for spine in spines:
        axes.spines[spine].set_visible(False)
    return axes



#### DAILY REPORT FUNCTONS ####
def ephys_report(df_ephys_behavior, behavior_sorted_columns, save_path):
    print('Generating Ephys Report...')

    ####  INITIAL PARAMETERS ####

    # Create a sorted behaviral dataframe (useful for raster plots)
    behavior_sorted_columns_copy= behavior_sorted_columns.copy()
    behavior_sorted_columns_copy.remove('trial')
    behavior_sorted_columns_copy.append('trial_index_response')
    df_behavior_sorted = df_ephys_behavior.groupby('trial')[behavior_sorted_columns_copy].max().reset_index()

    # Latencies between events
    diff_corridor = (df_behavior_sorted['corridor3'] - df_behavior_sorted['corridor1']).tolist()
    diff_corridor_response = (df_behavior_sorted['response'] - df_behavior_sorted['corridor3']).tolist()
    diff_response_end = (df_behavior_sorted['STATE_Exit_END_align'] - df_behavior_sorted['response']).tolist()
    trial_index_list = df_behavior_sorted['trial_index_response'].values.tolist()

    # Colors and labels to plot
    stim_o, stim_f, stim_c = [-1, 0, 1], ['Left', 'Centre', 'Right'], [(0.88, 0.43, 0.46), (0.56, 0.82, 0.55), (0.45, 0.62, 0.74)]
    tresult_o, tresult_f, tresult_c = ['correct_first', 'punish'], ['Correct', 'Incorrect'], ['lightseagreen', 'black']

    # Page specifications
    units_per_pag, alignment_event = 2, [
        {'name': 'corridor1', 'align':'align_corridor1', 'boundaries': [-1.2, 0.8], 'x_pos': 0,  'colspan': 13, 'title_pos': 0.6, 'first': True, 'header':False, 'linesxl':utils.change_sign_list(diff_response_end), 'linesxr':diff_corridor, 'linesy':trial_index_list},
        {'name': 'corridor3', 'align':'align_corridor3', 'boundaries': [-0.8, 0.8], 'x_pos': 14, 'colspan': 9, 'title_pos': 0.5, 'first': False, 'header':True, 'linesxl':utils.change_sign_list(diff_corridor), 'linesxr':diff_corridor_response, 'linesy':trial_index_list},
        {'name': 'response',  'align':'align_response',  'boundaries': [-0.8, 1.2], 'x_pos': 24, 'colspan': 13, 'title_pos': 0.4, 'first': False, 'header':False, 'linesxl':utils.change_sign_list(diff_corridor_response), 'linesxr':diff_response_end, 'linesy':trial_index_list}]
    bin_size, sdev, autocorr_win, autocorr_x = 1, 50, 60, 39 # Convolution parameters (ms)

    units = df_ephys_behavior.cluster_id.unique()
    units.sort()


    ###########  GENERATE PDF ###########
    with PdfPages(save_path) as pdf:
        for i in range(0, len(units), units_per_pag): # LOOP THROUGH UNITS BY PAGE (i: page index)
            plt.figure(figsize=(15, 11.7)) # CREATE A4 PAGE

            for j, unit in enumerate(units[i:i + units_per_pag]): # LOOP THOUGH UNITS INSIDE THE PAGE (j: unit index inside the page)
                unit_df = df_ephys_behavior[df_ephys_behavior.cluster_id == unit] # SELECT UNIT
                y_pos = 3 if j == 0 else 27 # SELECT POSITION INSIDE THE PAGE

                # CALCULATE FIRING RATE
                rounded_mfr, spiketrain = utils.calculate_firing_rate(unit_df)

                # HEADER
                unit_type = unit_df['group'].max()
                subject = unit_df.subject.max()
                date = unit_df.date.max()
                hemisphere = unit_df.hemisphere.max()
                header = f'Subject: {subject} ; Date: {date} ; Hemisphere: {hemisphere} ; Unit: {int(unit)} ; Type: {unit_type} ; Mean fr: {rounded_mfr} Spikes/s'

                # AXES LIMITS
                y_max_trial= unit_df.trial.max() + 10

                # CONVOLUTION
                conv_df = utils.generate_conv_data(spiketrain, bin_size, sdev)

                # GENERATE CONVOLUTED DATAFRAME
                aligned_events_list = []   # Align conv_times with df_behavior_sorted based on timestamps_fix_min and timestamps_fix_max
                for _, row in df_behavior_sorted.iterrows():
                    classified_times = utils.classify_time(row, conv_df)
                    classified_times['trial'] = row['trial']
                    aligned_events_list.append(classified_times)
                conv_aligned_df = pd.concat(aligned_events_list)
                df_beh_conv = df_behavior_sorted.merge(conv_aligned_df, on='trial', how='left') # Merge aligned data with df_trial

                # INCLUDE ALIGNED EVENTS IN DF_CONV & CALCULATE YMAX FOR PSTH
                y_max_fr = 0
                for event in alignment_event:
                    df_beh_conv[event['align']] = round(df_beh_conv['conv_times'] - (df_beh_conv['corridor_ttl_fix'] + df_beh_conv[event['name']]),3)  # ALIGN TIMESTAMPS TO TTL COLUMNS
                    unit_df[event['align']] = round(unit_df['timestamps_fix'] - (unit_df['corridor_ttl_fix'] + unit_df[event['name']]), 3)  # ALIGN TIMESTAMPS TO TTL COLUMNS
                    to_plot = df_beh_conv.loc[(df_beh_conv[event['align']] >= event['boundaries'][0]) &
                                              (df_beh_conv[event['align']] <= event['boundaries'][1])]
                    to_plot = to_plot.groupby([event['align'], 'r_n'])['conv_firing'].mean().reset_index()
                    y_max_fr = max(y_max_fr, to_plot['conv_firing'].max())

                # LOOP THOUGH EVENTS
                for event in alignment_event:
                    # PLOT RASTER
                    axes = plot_raster(event['x_pos'], y_pos,  event['colspan'], unit_df, event,  'r_n', y_max_trial,
                                       event['title_pos'], stim_o, stim_c, stim_f, event['linesxl'],  event['linesxr'], event['linesy'])
                    if event['header']:
                        axes.text(0.5, 1.25, header, ha='center', fontweight='bold', transform=axes.transAxes,
                                  fontsize=10)

                    # PLOT PSTH
                    plot_psth(event['x_pos'], y_pos + 10, event['colspan'], df_beh_conv, event, 'r_n', y_max_fr,  stim_o,
                          stim_c, stim_f)
                    if event['name'] == 'response': # plot an extra response PSTH splitted by outcome
                        new_event = event.copy()
                        new_event['name'] = 'response_reward'
                        new_event['boundaries'] = [-0.2, 1.8]
                        new_event['x_pos'] = autocorr_x-1
                        plot_psth(new_event['x_pos'] , y_pos + 10, new_event['colspan'], df_beh_conv, new_event, 'trial_result', y_max_fr, tresult_o,
                                  tresult_c, tresult_f)
                #PLOT AUTOCORRELOGRAM
                bins_auto, autocorr_array = utils.generate_autocorr_data(spiketrain, bin_size+1, autocorr_win)
                plot_autocorr(autocorr_x,  y_pos, bins_auto, autocorr_array)

            ########### SAVING AND CLOSING PAGE ###########
            plt.tight_layout()
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
            pdf.savefig(dpi=50)  # Lower resolution
            plt.close()
        print('New ephys report completed successfully :)')







