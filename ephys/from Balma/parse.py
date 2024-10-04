import os
import numpy as np
import pandas as pd
import utils
import plots
import warnings
warnings.filterwarnings('ignore')


def parse_behavior(df, states_list):
    # Apply conversion to valid columns
    df = utils.convert_strings_to_lists(df, ['Port4In_START'])

    ## CREATE LAST FIXATION1 COLUMN (when animal corsses for the last time the begining of the corridr)
    df['STATE_Fixation1_START_last'] = df.apply(utils.find_last_before, x='Port4In_START',  y='STATE_Fixation2_START',
                                                z='STATE_Fixation1_START',   axis=1)

    ## CATEGORIZE STIMULUS & RESPONSE POSITIONS FOR 3 CHOICES
    #  stimulus positions
    df['x_n'] = np.nan
    bins_x = [65.0, 200.0, 335.0]
    cateogries = [-1, 0, 1]
    df['x_n'] = df['x'].replace(bins_x, cateogries)
    #  response positions
    bins_r = pd.IntervalIndex.from_tuples([(20, 100), (160, 240), (300, 380)])  # 3 choices
    df['response_x'] = pd.to_numeric(df['response_x'], errors='coerce')  # correct number format
    df['r_n'] = np.nan
    df['r_n'] = pd.cut(df.response_x, bins=bins_r, ordered=True).map(dict(zip(bins_r, cateogries)))
    df['r_n'] = df['r_n'].astype(float)  # convert to numerical value (from categorical)

    ## INCLUDE COLUMNS OF INTEREST
    df['corridor1'] = df['STATE_Fixation1_START_last'] - df['STATE_Fixation1_START']
    df['corridor2'] = df['STATE_Fixation2_START'] - df['STATE_Fixation1_START']
    df['corridor3'] = df['STATE_Pre_Response_window_START'] - df['STATE_Fixation1_START']
    df['response'] =  df['STATE_Response_window_END'] - df['STATE_Fixation1_START']
    df['stim_dur_ext'] = df.apply(utils.classify_stim_dur_ext, axis=1)

    ## ALIGN TO THE BEGINNING OF THE SESSION
    valid_columns = [column for column in states_list if column in df.columns] # Remove columns not present in the df
    starting_time = df['STATE_Start_task_START'].iloc[0]  # starting time of the session
    states_list_aligned = []
    for idx, state in enumerate(valid_columns):
        new_col_name = str(state) + '_align'
        df[state] = pd.to_numeric(df[state], errors='coerce') # necessary to be able to align columns with nans
        df[new_col_name] = df[state] - starting_time
        states_list_aligned.append(new_col_name)

    return df, starting_time, states_list_aligned



def parse_ttls(samples, channels_of_interest, timestamps, bitsVolts, sampling_freq):
    """ Retruns.........."""

    # Get data for specific channels
    ttl_samples = utils.get_channel_data(samples, channels_of_interest)

    # Create name for the channels
    ch_names = []
    for ch in channels_of_interest:
        name = 'ADC' + str(ch)
        ch_names.append(name)

    # Create an initial dataframe
    df_ttl = pd.DataFrame({
        ch_names[0]: ttl_samples[16],
        ch_names[1]: ttl_samples[17],
        'timestamps': timestamps})

    # Convert bits to volts
    df_ttl['ADC16_V'] = df_ttl['ADC16'] * bitsVolts
    df_ttl['ADC17_V'] = df_ttl['ADC17'] * bitsVolts

    # Recover the first timestamp of the session.
    # This is important because sometimes it is not 0 if other animals were recorded before.
    initial = df_ttl.timestamps.iloc[0]
    # Divide timestamps by sampling frequency and also assuming that the first spike is the beginning of the ephys session
    df_ttl['timestamps_fix'] = (df_ttl.timestamps - initial) / sampling_freq

    ## TTLs voltage fluctuate form 0 to 4.68. There are few intermediate values. I selected the threshold >4 after observing false alarms in lower values
    # Transform the analog channels to boolean (Values fluctuate from 0 to 4.68V, we consider real signal above 3V.
    df_ttl['TTL1_bool'] = 0
    df_ttl.loc[df_ttl['ADC16_V'] >= 4.5, 'TTL1_bool'] = 1

    df_ttl['TTL2_bool'] = 0
    df_ttl.loc[df_ttl['ADC17_V'] >= 4.5, 'TTL2_bool'] = 1

    # Identify when there  they turn OFF/ON
    df_ttl['diff_TTL1'] = df_ttl.TTL1_bool.diff()  # +1 of turns ON -1 if turns OFF
    df_ttl['diff_TTL2'] = df_ttl.TTL2_bool.diff()

    # Check TTLs duration
    ttl_min_clock, ttl_max_clock = min(df_ttl['timestamps_fix']), max(df_ttl['timestamps_fix'])

    # Get only the timestamps in which there is a change in TTLs
    df_ttl = df_ttl.loc[(df_ttl['diff_TTL1'] == 1) | (df_ttl['diff_TTL1'] == -1) |
                        (df_ttl['diff_TTL2'] == 1) | (df_ttl['diff_TTL2'] == -1)]

    # SPLIT EVENTS OF TTLS
    # Response time: when the 2 event happend at the same time
    df_ttl['response_ttl'] = np.where((df_ttl['TTL1_bool'] == 1) & (df_ttl['TTL2_bool'] == 1), 1, 0)
    # Corridor: when the TTL2 alone is sent
    df_ttl['corridor_ttl'] = np.where((df_ttl['TTL2_bool'] == 1) & (df_ttl['TTL1_bool'] == 0), 1, 0)
    # Delay: when the TTL2 alone is sent (never in VG)
    df_ttl['delay_ttl'] = np.where((df_ttl['TTL2_bool'] == 0) & (df_ttl['TTL1_bool'] == 1), 1, 0)

    return df_ttl, ttl_min_clock, ttl_max_clock



def parse_spikes(spike_times, spike_clusters, df_labels, sampling_freq):

    # Transforms array of lists into array of ints
    spike_times_list = [item for sublist in spike_times for item in sublist]

    # Put the data in a single dataframe, one colum spikes, one colum clusters
    spike_times_serie = pd.Series(spike_times_list, name='times')
    spike_clusters_serie = pd.Series(spike_clusters, name='cluster_id')
    df_spikes = pd.concat([spike_times_serie, spike_clusters_serie], axis=1)

    # Merge with cluster labels, use cluster ID to associate each one
    df_spikes = pd.merge(df_spikes, df_labels, on=['cluster_id'])

    # Select only clusters labelled good or mua, which are presumably single units or mua
    df_spikes = df_spikes.loc[(df_spikes.group == 'good') | (df_spikes.group == 'mua')]

    # Transform the values per session to seconds. This takes into account the framerate of the recordings, 30000Hz for all the sessions.
    df_spikes['timestamps_fix'] = (df_spikes.times / sampling_freq)
    spike_min_clock, spike_max_clock = min(df_spikes['timestamps_fix']), max(df_spikes['timestamps_fix'])

    return df_spikes, spike_min_clock, spike_max_clock


def alignment(df_behavior, df_ttl, states_list_aligned):
    # Take  ttl arrays
    corridor_ttl = df_ttl.loc[df_ttl['corridor_ttl'] == 1]['timestamps_fix'].values
    response_ttl = df_ttl.loc[df_ttl['response_ttl'] == 1]['timestamps_fix'].values
    delay_ttl = df_ttl.loc[df_ttl['delay_ttl'] == 1]['timestamps_fix'].values

    # Align timestamps with first event corridor
    alignment_diff_c=corridor_ttl[0] -df_behavior['STATE_Fixation1_START_align'].iloc[0]
    if alignment_diff_c<0:
        print('WARNING! BEHAVIORAL CLOCK STARTING BEFORE EPHYS!')

    # TTL clock should start before the behavior, TTL times are longer. Add time to the behavior
    for idx, state in enumerate(states_list_aligned):
        new_col_name = str(state)
        df_behavior[new_col_name] = df_behavior[state] + alignment_diff_c

    # Include ttls in the behavioral df
    df_behavior['corridor_ttl'] = df_behavior.apply(lambda x: corridor_ttl[(corridor_ttl >= x['STATE_Start_task_START_align'])
                               & (corridor_ttl < x['STATE_Exit_END_align'])].tolist(), axis=1)
    df_behavior['delay_ttl'] = df_behavior.apply(lambda x: delay_ttl[(delay_ttl >= x['STATE_Start_task_START_align'])
                              & (delay_ttl < x['STATE_Exit_END_align'])].tolist(), axis=1)
    df_behavior['response_ttl'] = df_behavior.apply(lambda x: response_ttl[(response_ttl >= x['STATE_Start_task_START_align'])
                               & (response_ttl < x['STATE_Exit_END_align'])].tolist(), axis=1)

    # If multiple ttl per trial select the closest
    df_behavior['corridor_ttl_fix'] =df_behavior.apply(lambda x: utils.select_closest(x['corridor_ttl'], x['STATE_Fixation1_START_align']), axis=1)
    df_behavior['response_ttl_fix'] =df_behavior.apply( lambda x:utils.select_closest(x['response_ttl'], x['STATE_Response_window_END_align']), axis=1)
    df_behavior['delay_start'] = df_behavior.apply(lambda x: x['STATE_Pre_Response_window_START_align'] if x['trial_type'] == 'WM_I' else (
            x['STATE_Fixation2_START_align'] if x['trial_type'] == 'WM_D' else np.nan), axis=1)
    df_behavior['delay_ttl_fix'] =  df_behavior.apply(lambda x: utils.select_closest(x['delay_ttl'], x['delay_start']), axis=1)

    return df_behavior, alignment_diff_c


def merging(df_behavior, behavior_sorted_columns, df_spikes, hemisphere):

    # Create sorted df
    df_spikes_sorted = df_spikes.sort_values(by=['timestamps_fix'])
    df_spikes_sorted = df_spikes_sorted[['cluster_id', 'group', 'timestamps_fix']]

    # Check trials without ttls
    missing_corridor_ttl = df_behavior.loc[df_behavior['corridor_ttl_fix'].isna(), 'trial'].values
    missing_response_ttl = df_behavior.loc[df_behavior['response_ttl_fix'].isna(), 'trial'].values
    missing_delay_ttl = df_behavior.loc[df_behavior['delay_ttl_fix'].isna(), 'trial'].values

    # Drop trials without corridor ttl (our main aligner)
    if len(missing_corridor_ttl) > 0:
        df_behavior = df_behavior.dropna(subset=['corridor_ttl_fix'])
        print('WARNING REMOVED TRIALS: ' + str(missing_corridor_ttl))

    # Add new columns to df_spikes
    df_spikes_sorted['trial'] = np.nan
    df_spikes_sorted['subject'] = df_behavior['subject'].iloc[0]
    df_spikes_sorted['date'] = df_behavior['date'].iloc[0]
    df_spikes_sorted['hemisphere'] = hemisphere

    # Assign trial numbers based on the intervals defined in df_behavior
    for i in range(df_behavior.trial.nunique()):
        start_time = df_behavior['STATE_Start_task_START_align'].iloc[i]
        end_time = df_behavior['STATE_Exit_END_align'].iloc[i]
        trial_num = df_behavior['trial'].iloc[i]
        mask = (df_spikes_sorted['timestamps_fix'] >= start_time) & (df_spikes_sorted['timestamps_fix'] < end_time)
        df_spikes_sorted.loc[mask, 'trial'] = trial_num

    # Assign NaN to timestamps before the first trial and after the last trial
    first_trial_start_time = df_behavior['STATE_Start_task_START_align'].iloc[0]
    last_trial_end_time = df_behavior['STATE_Exit_END_align'].iloc[-1]
    df_spikes_sorted.loc[df_spikes_sorted['timestamps_fix'] < first_trial_start_time, 'trial'] = np.nan
    df_spikes_sorted.loc[df_spikes_sorted['timestamps_fix'] > last_trial_end_time, 'trial'] = np.nan

    # Merge completely behavior with spikes
    df_ephys_behavior = df_spikes_sorted.merge(df_behavior[behavior_sorted_columns], on=['subject', 'date', 'trial'], how='left')

    # Round time columns to 3 decimals (ms) to save space
    to_round = ['corridor1', 'corridor2', 'corridor3', 'response', 'stim_dur_ext',
                'corridor_ttl_fix', 'delay_ttl_fix', 'response_ttl_fix', 'timestamps_fix']
    df_ephys_behavior[to_round] = df_ephys_behavior[to_round].round(3)


    # Create a new colum with trials index ordered by choice
    df_trial_rn = df_ephys_behavior.groupby('trial')['r_n'].max().reset_index()
    df_sorted = df_trial_rn.sort_values(by=['r_n', 'trial'], ascending=[True, True])
    df_sorted['trial_index_response'] = range(1, len(df_sorted) + 1)
    df_sorted.loc[df_sorted['r_n'].isnull(), 'trial_index_response'] = np.nan
    df_ephys_behavior = df_ephys_behavior.merge(df_sorted[['trial', 'trial_index_response']], on='trial',
                                                how='left') # Merge df_ephys_behavior with df_sorted on the 'trial' column

    return df_ephys_behavior, missing_corridor_ttl, missing_response_ttl, missing_delay_ttl

