# To do:

# Include the new VARs in upper text

########################################################################################################################

# Import modules
import pandas as pd
import numpy as np


########################################################################################################################

def parse_v2(path):

    # Don't take first 6 lines (they start with __underscores__ and it crashes)
    df = pd.read_csv(path, skiprows=6, sep=';')
    sounds_2 = pd.read_csv('/home/alexis/PycharmProjects/create_sounds/sounds_2.csv')

    ####################################################################################################################

    # index = filenames[filenames['TYPE'] == 'END-TRIAL'].index
    index = df[df['TYPE'] == 'TRIAL'].index  # Use this one because after END-TRIAL it comes the summary of the previous
    # one
    n_trials = len(index) - 1  # Number of trials (= i + 1)

    # INFO (METADATA)
    serial_port = [df[df.MSG == 'SERIAL-PORT']['+INFO'].iloc[0]] * n_trials
    protocol = [df[df.MSG == 'PROTOCOL-NAME']['+INFO'].iloc[0]] * n_trials  # Task
    creator = df[df.MSG == 'CREATOR-NAME']['+INFO'].iloc[0]
    project = [df[df.MSG == 'PROJECT-NAME']['+INFO'].iloc[0]] * n_trials
    experiment = [df[df.MSG == 'EXPERIMENT-NAME']['+INFO'].iloc[0]] * n_trials
    board = [df[df.MSG == 'BOARD-NAME']['+INFO'].iloc[0]] * n_trials  # Box
    setup = [df[df.MSG == 'SETUP-NAME']['+INFO'].iloc[0]] * n_trials
    net_port = [df[df.MSG == 'NET-PORT']['+INFO'].iloc[0]] * n_trials
    subject = df[df.MSG == 'SUBJECT-NAME']['+INFO'].iloc[0]
    bpod_api_version = [df[df.MSG == 'BPOD-API-VERSION']['+INFO'].iloc[0]] * n_trials
    session = [df[df.MSG == 'SESSION-NAME']['+INFO'].iloc[0]] * n_trials
    session_started = df[df.MSG == 'SESSION-STARTED']['+INFO'].iloc[0]
    # If the session ends abruptly due to an error, 'SESSION-ENDED' won't show up. Use 'PC-TIME' instead
    try:
        session_ended = df[df.MSG == 'SESSION-ENDED']['+INFO'].iloc[0]
    except IndexError:
        session_ended = df[df.TYPE == 'EVENT']['PC-TIME'].iloc[-1]
        print(f"The session '{np.unique(session)[0]}' ended abruptly. Using last PC_TIME "
              f"timestamp instead")

    # VAL
    # Bpod's VARs
    aw = [int(str(df[df.MSG == 'VAR_AW']['+INFO'].iloc[0]))] * n_trials
    switch = [float(str(df[df.MSG == 'VAR_SWITCH']['+INFO'].iloc[0]))] * n_trials
    timeout = [float(str(df[df.MSG == 'VAR_TIMEOUT']['+INFO'].iloc[0]))] * n_trials
    fixation = [float(str(df[df.MSG == 'VAR_FIXATION']['+INFO'].iloc[0]))] * n_trials
    stage = [int(str(df[df.MSG == 'VAR_STAGE']['+INFO'].iloc[0]))] * n_trials
    # substage = [int(str(df[df.MSG == 'VAR_SUBSTAGE']['+INFO'].iloc[0]))] * n_trials
    motor = [int(str(df[df.MSG == 'VAR_MOTOR']['+INFO'].iloc[0]))] * n_trials
    rec = [int(str(df[df.MSG == 'VAR_REC']['+INFO'].iloc[0]))] * n_trials
    progression = [int(str(df[df.MSG == 'VAR_PROGRESSION']['+INFO'].iloc[0]))] * n_trials
    cb = [int(float(str(df[df.MSG == 'VAR_CB']['+INFO'].iloc[0])))] * n_trials
    resp_win = [int(str(df[df.MSG == 'VAR_RESP_WIN']['+INFO'].iloc[0]))] * n_trials
    iti = [int(str(df[df.MSG == 'VAR_ITI']['+INFO'].iloc[0]))] * n_trials
    warm_up = [int(str(df[df.MSG == 'VAR_WARM_UP']['+INFO'].iloc[0]))] * n_trials
    recovery_mode = [int(str(df[df.MSG == 'VAR_RECOVERY_MODE']['+INFO'].iloc[0]))] * n_trials
    try:
        p_right = [float(df[df.MSG == 'VAR_P_RIGHT']['+INFO'].iloc[0])] * n_trials  # Added on 06-04-2022
    except IndexError:
        p_right = [np.nan] * n_trials

    try:
        blocks = [(df[df.MSG == 'VAR_BLOCKS']['+INFO'].iloc[0])] * n_trials  # Added on 08-03-2023
    except IndexError:
        blocks = [np.nan] * n_trials

    try:
        stim_dur = [float(df[df.MSG == 'VAR_STIM_DUR']['+INFO'].iloc[0])] * n_trials  # Added on 14-03-2023
    except IndexError:
        stim_dur = [np.nan] * n_trials

    try:
        delay = [float(df[df.MSG == 'VAR_DELAY']['+INFO'].iloc[0])] * n_trials  # Added on 14-03-2023
    except IndexError:
        delay = [np.nan] * n_trials

    try:
        block_len = [float(df[df.MSG == 'VAR_BLOCK_LEN']['+INFO'].iloc[0])] * n_trials  # Added on 31-03-2023
    except IndexError:
        block_len = [np.nan] * n_trials


    # Registered values (out of loop)
    reward_side = df[df.MSG == 'REWARD_SIDE']['+INFO'].iloc[-1]  # [-1] to take the last one in case CB was on
    reward_side_original = df[df.MSG == 'REWARD_SIDE']['+INFO'].iloc[0]  # [0] to take the first one in case CB was on

    # # under testing: objective is to plot the cb trials in daily report
    # reward_side_cb = df[df.MSG == 'REWARD_SIDE']['+INFO']
    # print(reward_side_cb)

    valve_1 = [float(df[df.MSG == 'VALVE_1']['+INFO'].iloc[0])] * n_trials
    valve_2 = [float(df[df.MSG == 'VALVE_2']['+INFO'].iloc[0])] * n_trials

    # Data curation
    creator = [creator.split()[0][2:-2]] * n_trials
    subject = [subject.split()[0][2:-2]] * n_trials
    date = [session_started.split()[0]] * n_trials
    time_session_started = [session_started.split()[1]] * n_trials
    time_session_ended = [session_ended.split()[1]] * n_trials
    reward_side = reward_side[1:-1].split(',')  # Convert string to list, [1:-1] to get rid of the square brackets []
    reward_side = list(map(int, reward_side))  # Convert list elements from string to integers
    # reward_side = np.array(reward_side, dtype=int)  # Convert to array
    reward_side = reward_side[:n_trials]

    # Added on 02.04.2023
    reward_side_original = reward_side_original[1:-1].split(',')  # Convert string to list, [1:-1] to get rid of the square brackets []
    reward_side_original = list(map(int, reward_side_original))  # Convert list elements from string to integers
    # reward_side_original = np.array(reward_side_original, dtype=int)  # Convert to array
    reward_side_original = reward_side_original[:n_trials]

    # Trials in which CB was activated. Should capture the trials in which the reward side was changed
    cb_trials = cb_trials = [1 if reward_side[i] != reward_side_original[i] else 0 for i in range(len(reward_side))]

    ####################################################################################################################

    trial = []
    rep_trial = []
    reward = []
    punish = []
    miss = []
    wrong_lick = []
    hit = []
    after_hit = []
    choice = []
    rep_choice = []
    response = []
    trial_start = []
    trial_end = []
    trial_len = []
    stim_start = []
    stim_end = []
    stim_len = []
    resp_win_start = []
    resp_win_end = []
    resp_win_len = []
    filename = []
    filename2 = []
    files_match = []
    message = []
    message_found = []
    sound_left = []
    sound_right = []
    sound = []
    # evidence = []
    # evi_rep = []
    # coh_rep = []
    # coherence = []
    ild = []
    ild_rep = []
    port1in = []
    port1out = []
    port2in = []
    port2out = []
    # substage = []
    p = []

    # Added on 02.04.2023, but it should have 100% backwards compatibility
    aw_trials = []  # Trials in which AW was given. Should capture the initial AW trials plus the ones as CB measure

    ####################################################################################################################

    for i in range(len(index) - 1):  # -1 to not take into account last trial

        trial.append(i)
        band = df[index[i]:index[i + 1]]

        # Rewards
        if not pd.isnull(band[(band.TYPE == 'STATE') & (band.MSG == 'Reward')]['+INFO'].iloc[0]):
            reward.append(1)
        else:
            reward.append(0)

        # Punishes
        if not pd.isnull(band[(band.TYPE == 'STATE') & (band.MSG == 'Punish')]['+INFO'].iloc[0]):
            punish.append(1)
        else:
            punish.append(0)

        # Misses/Responses
        if not pd.isnull(band[(band.TYPE == 'STATE') & (band.MSG == 'Miss')]['+INFO'].iloc[0]):
            miss.append(1)
            response.append(0)
        else:
            miss.append(0)
            response.append(1)

        # WrongLicks
        try:
            if not pd.isnull(band[(band.TYPE == 'STATE') & (band.MSG == 'WrongLick')]['+INFO'].iloc[0]):
                wrong_lick.append(1)
            else:
                wrong_lick.append(0)
        except IndexError:
            wrong_lick.append(np.nan)

        if miss[i] == 1:
            hit.append(np.nan)
            choice.append(np.nan)
        else:
            hit.append(reward[i])
            choice.append(1) if reward_side[i] == hit[i] else choice.append(0)
            # Another way is to check PortIns

        if wrong_lick[i] == 1:  # Consider wrong licks as punish
            hit[i] = 0

        # Get all trials with AW, including those as counter bias measure (02.04.2023):
        if pd.isnull(float(band[(band['TYPE'] == 'STATE') & (band['MSG'] == 'AW')]['+INFO'].iloc[0])):
            aw_trials.append(0)  # If it's nan AW state was not visited
        else:
            aw_trials.append(1)  # Else Aw was given

        # Trial timestamps
        trial_start.append(band[band.TYPE == 'INFO']['BPOD-INITIAL-TIME'].iloc[0])
        trial_end.append(band[band.TYPE == 'INFO']['BPOD-FINAL-TIME'].iloc[0])
        trial_len.append(float(band[band.TYPE == 'INFO']['+INFO'].iloc[0]))

        # Stimulus timestamps (take BPOD-FINAL-TIME as the real timestamp of the TTL = BPOD-INITIAL-TIME + TTL duration)
        stim_start.append(
            float(band[(band['TYPE'] == 'STATE') & (band['MSG'] == 'StimulusTrigger')]['BPOD-FINAL-TIME'].iloc[0]))

        # This if block is because the finite state machine only goes over 'StimulusStop' after a Hit
        if stage[i] <= 3:
            if miss[i] == 1:
                stim_end.append(
                    float(band[(band['TYPE'] == 'STATE') & (band['MSG'] == 'Miss')]['BPOD-FINAL-TIME'].iloc[0]))
            elif punish[i] == 1:
                stim_end.append(
                    float(band[(band['TYPE'] == 'STATE') & (band['MSG'] == 'Punish')]['BPOD-FINAL-TIME'].iloc[0]))
            else:  # Reward or WrongLick
                stim_end.append(
                    float(band[(band['TYPE'] == 'STATE') & (band['MSG'] == 'StimulusStop')]['BPOD-FINAL-TIME'].iloc[0]))
        else:  # Leads to erroneous stimulus length in stage 4, but probably was there for stage 1. Readjust if needed
            stim_end.append(
                float(band[(band['TYPE'] == 'STATE') & (band['MSG'] == 'StimulusStop')]['BPOD-FINAL-TIME'].iloc[0]))

        # # Because all sounds in batches 1-3 (sounds.csv/sounds_2.csv) are 1 s
        # if stim_end[i] - stim_start[i] > 1:
        #     stim_len.append(1)
        # else:
        #     stim_len.append(stim_end[i] - stim_start[i])

        stim_len.append(stim_end[i] - stim_start[i])

        # Response window
        resp_win_start.append(
            band[(band['TYPE'] == 'STATE') & (band['MSG'] == 'ResponseWindow')]['BPOD-INITIAL-TIME'].iloc[0])
        resp_win_end.append(
            band[(band['TYPE'] == 'STATE') & (band['MSG'] == 'ResponseWindow')]['BPOD-FINAL-TIME'].iloc[0])
        resp_win_len.append(float(band[(band['TYPE'] == 'STATE') & (band['MSG'] == 'ResponseWindow')]['+INFO'].iloc[0]))

        # Licks timestamps
        port1in.append(band[band['+INFO'] == 'Port1In']['BPOD-INITIAL-TIME'].tolist())
        port1out.append(band[band['+INFO'] == 'Port1Out']['BPOD-INITIAL-TIME'].tolist())
        port2in.append(band[band['+INFO'] == 'Port2In']['BPOD-INITIAL-TIME'].tolist())
        port2out.append(band[band['+INFO'] == 'Port2Out']['BPOD-INITIAL-TIME'].tolist())

        # Registered values (within loop)
        filename.append(band[(band['TYPE'] == 'VAL') & (band['MSG'] == 'FILENAME')]['+INFO'].iloc[0])  # Bpod sounds

        # SOUND CHECKS (not registered from the beginning except Filename)
        # Filename2 (registered by Arduino)
        try:
            filename2.append(band[(band['TYPE'] == 'VAL') & (band['MSG'] == 'FILENAME2')]['+INFO'].iloc[0])  # Arduino
            # sounds
        except IndexError:
            filename2.append(np.nan)

        if filename[i] == filename2[i]:
            files_match.append(1)
        elif filename2[i] is np.nan:
            files_match.append(0)
        else:
            files_match.append(0)

        # Arduino's error messages
        try:
            message.append(str(band[(band['TYPE'] == 'VAL') & (band['MSG'] == 'MESSAGE')]['+INFO'].iloc[0]))
        except IndexError:
            message.append(None)  # As when there was no message it gets filled with nan and I want to know when I
            # started to keep track of it

        if message[i] is None:
            message_found.append(np.nan)
        elif message[i] == 'nan':
            message_found.append(0)
        else:
            message_found.append(1)

        # Sound detection with Albert's board
        # Sound from left
        if band[(band['TYPE'] == 'EVENT') & (band['+INFO'] == 'BNC1High')].shape[0] > 0:
            sound_left.append(1)
        else:
            sound_left.append(0)

        # Sound from right
        if band[(band['TYPE'] == 'EVENT') & (band['+INFO'] == 'BNC2High')].shape[0] > 0:
            sound_right.append(1)
        else:
            sound_right.append(0)

        sound.append(sound_left[i] + sound_right[i])

        # evidence.append(float(band[(band['TYPE'] == 'VAL') & (band['MSG'] == 'EVIDENCE')]['+INFO'].iloc[0]))
        # coherence.append(float(band[(band['TYPE'] == 'VAL') & (band['MSG'] == 'COHERENCE')]['+INFO'].iloc[0]))
        try:  # In case sound parameters are lost in the void, they can always be rescued from their csv
            ild.append(float(band[(band['TYPE'] == 'VAL') & (band['MSG'] == 'ILD')]['+INFO'].iloc[0]))
        except IndexError:
            ild.append(sounds_2.loc[sounds_2.filename == filename[i]].ILD.values[0])
        # Sound filename + sound2 filename + coherence/evidence + presented coherences/evidences
        # Stage, motor, substage for tracking changes within session when running a single script
        # Register running window
        # substage.append(int(band[(band['TYPE'] == 'VAL') & (band['MSG'] == 'SUBSTAGE')]['+INFO'].iloc[0]))

        try:
            p.append(float(band[(band['TYPE'] == 'VAL') & (band['MSG'] == 'P')]['+INFO'].iloc[0]))
        except IndexError:
            p.append(np.nan)

        if i == 0:
            after_hit.append(np.nan)
            rep_choice.append(np.nan)
            rep_trial.append(np.nan)
            ild_rep.append(np.nan)
            # evi_rep.append(np.nan)
            # coh_rep.append(np.nan)

        if i > 0:  # All the following analysis can't be done in the first trial

            # After correct
            after_hit.append(hit[i - 1])

            # Repeat choice
            if np.isnan(choice[i - 1]) or np.isnan(choice[i]):
                rep_choice.append(np.nan)
            elif choice[i - 1] == choice[i]:
                rep_choice.append(1)
            else:
                rep_choice.append(0)

            # Trials in which the rewarded side coincides with the previous animal’s response (not rewarded side)
            if np.isnan(choice[i - 1]):
                rep_trial.append(np.nan)
            elif reward_side[i] == choice[i - 1]:
                rep_trial.append(1)  # Repeat trials
            else:
                rep_trial.append(0)  # Alternated trials

            # Evidence/coherence/ild repeat
            if np.isnan(rep_trial[i]):
                ild_rep.append(np.nan)
                # evi_rep.append(np.nan)
                # coh_rep.append(np.nan)
            elif rep_trial[i] == 0:  # Negative evi/coh
                ild_rep.append(-abs(ild[i]))
                # evi_rep.append(-abs(evidence[i]))
                # coh_rep.append(-abs(coherence[i]))
            elif rep_trial[i] == 1:  # Positive evi/coh
                ild_rep.append(abs(ild[i]))
                # evi_rep.append(abs(evidence[i]))
                # coh_rep.append(abs(coherence[i]))

    # If no sound was detected in an entire session or most trials (95%), it's likely due to Albert's card wasn't
    # installed. First sessions with it (in boxes 5-6) was on 27/10/2021
    # if len(np.unique(sound_left)) == 1 and len(np.unique(sound_right)) == 1:  # Sensitive to noise (1 trial with BNCXHigh)
    if len(np.where(np.array(sound_left) == 1)[0]) / len(np.where(np.array(reward_side) == 0)[0]) <= 0.05 and \
            len(np.where(np.array(sound_right) == 1)[0]) / len(np.where(np.array(reward_side) == 1)[0]) <= 0.05:
        sound_left = [np.nan] * n_trials
        sound_right = [np.nan] * n_trials
        sound = [np.nan] * n_trials

    ####################################################################################################################

    # Construct DataFrame
    columns = ['Trial', 'Side', 'RepTrial', 'Reward', 'Punish', 'Miss', 'WrongLick', 'Hit', 'AfterHit', 'Choice',
               'RepChoice', 'Response', 'TrialStart', 'TrialEnd', 'TrialLen', 'StimStart', 'StimEnd', 'StimLen',
               'RespWinStart', 'RespWinEnd', 'RespWinLen', 'Filename', 'Filename2', 'FilesMatch', 'Message',
               'MessageFound', 'SoundLeft', 'SoundRight', 'Sound', 'ILD', 'ILDRep', 'Port1In', 'Port1Out', 'Port2In',
               'Port2Out', 'ValveLeft', 'ValveRight', 'AW', 'AWTrials', 'Switch', 'Timeout', 'Fixation', 'Stage', 'Motor', 'REC',
               'Progression', 'CB', 'RespWin', 'ITI', 'WarmUp', 'RecoveryMode', 'P', 'PRight', 'Blocks', 'BlockLen',
               'StimDur', 'Delay', 'SerialPort', 'Protocol', 'Creator', 'Project', 'Experiment', 'Board', 'Setup',
               'NetPort', 'Subject', 'BpodApiVersion', 'Session', 'Date', 'SessionStart', 'SessionEnd']

    data = list(zip(trial, reward_side, rep_trial, reward, punish, miss, wrong_lick, hit, after_hit, choice, rep_choice,
                    response, trial_start, trial_end, trial_len, stim_start, stim_end, stim_len, resp_win_start,
                    resp_win_end, resp_win_len, filename, filename2, files_match, message, message_found, sound_left,
                    sound_right, sound, ild, ild_rep, port1in, port1out, port2in, port2out, valve_1, valve_2, aw, aw_trials,
                    switch, timeout, fixation, stage, motor, rec, progression, cb, resp_win, iti, warm_up, recovery_mode,
                    p, p_right, blocks, block_len, stim_dur, delay, serial_port, protocol, creator, project, experiment,
                    board, setup, net_port, subject, bpod_api_version, session, date, time_session_started,
                    time_session_ended))

    df_session = pd.DataFrame(data=data, columns=columns)

    # df_session.to_csv(str('parsed_') + path.split('/')[-1])  # Save df_session as csv file

    return df_session

# if __name__ == "__main__":
#     parse()

# Debug
path = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_4/setups/561/sessions/561_stage_training_v4_20230329-110651' \
       '/561_stage_training_v4_20230329-110651.csv'
df = parse_v2(path)
