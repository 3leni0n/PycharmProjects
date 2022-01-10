import pandas as pd
import numpy as np
import time


def mini_parse(file):

    colnames = ['TYPE', 'PC-TIME', 'BPOD-INITIAL-TIME', 'BPOD-FINAL-TIME', 'MSG', '+INFO']
    df = pd.read_csv(file.path, skiprows=file.skip, sep=';', names=colnames, header=None)

    index = df[df['TYPE'] == 'TRIAL'].index

    if len(index) < 2:
        return

    if file.df is None:
        reward_side = df[df.MSG == 'REWARD_SIDE']['+INFO'].iloc[-1]  # [-1] to take the last one in case CB was on
        file.reward_side = reward_side
    else:
        reward_side = file.reward_side

    n_trials = len(index) - 1  # Number of trials (= i +1)

    # METADATA (multiply by n_trials)

    # # INFO
    # serial_port = [df[df.MSG == 'SERIAL-PORT']['+INFO'].iloc[0]] * n_trials
    # protocol = [df[df.MSG == 'PROTOCOL-NAME']['+INFO'].iloc[0]] * n_trials  # Task
    # creator = df[df.MSG == 'CREATOR-NAME']['+INFO'].iloc[0]
    # project = [df[df.MSG == 'PROJECT-NAME']['+INFO'].iloc[0]] * n_trials
    # experiment = [df[df.MSG == 'EXPERIMENT-NAME']['+INFO'].iloc[0]] * n_trials
    # board = [df[df.MSG == 'BOARD-NAME']['+INFO'].iloc[0]] * n_trials  # Box
    # setup = [df[df.MSG == 'SETUP-NAME']['+INFO'].iloc[0]] * n_trials
    # net_port = [df[df.MSG == 'NET-PORT']['+INFO'].iloc[0]] * n_trials
    # subject = df[df.MSG == 'SUBJECT-NAME']['+INFO'].iloc[0]
    # bpod_api_version = [df[df.MSG == 'BPOD-API-VERSION']['+INFO'].iloc[0]] * n_trials
    # session = [df[df.MSG == 'SESSION-NAME']['+INFO'].iloc[0]] * n_trials
    # session_started = df[df.MSG == 'SESSION-STARTED']['+INFO'].iloc[0]
    # # If the session ends abruptly due to an error, 'SESSION-ENDED' don't will show up
    # try:
    #     session_ended = df[df.MSG == 'SESSION-ENDED']['+INFO'].iloc[0]
    # except IndexError:
    #     # session_ended = df['PC-TIME'].iloc[-1]
    #     session_ended = df[df.TYPE == 'EVENT']['PC-TIME'].iloc[-1]
    #     print('The session ended abruptly, probably due to Bpod crashed. Using last PC_TIME timestamp instead')

    # VAL
    # Bpod's VARs
    # aw = [int(df[df.MSG == 'VAR_AW']['+INFO'].iloc[0])] * n_trials
    # switch = [float(df[df.MSG == 'VAR_SWITCH']['+INFO'].iloc[0])] * n_trials
    # timeout = [float(df[df.MSG == 'VAR_TIMEOUT']['+INFO'].iloc[0])] * n_trials
    # fixation = [float(df[df.MSG == 'VAR_FIXATION']['+INFO'].iloc[0])] * n_trials
    stage = [int(df[df.MSG == 'VAR_STAGE']['+INFO'].iloc[0])] * n_trials
    # substage = [int(df[df.MSG == 'VAR_SUBSTAGE']['+INFO'].iloc[0])] * n_trials
    # motor = [int(df[df.MSG == 'VAR_MOTOR']['+INFO'].iloc[0])] * n_trials
    # rec = [int(df[df.MSG == 'VAR_REC']['+INFO'].iloc[0])] * n_trials
    # progression = [int(df[df.MSG == 'VAR_PROGRESSION']['+INFO'].iloc[0])] * n_trials
    # cb = [int(df[df.MSG == 'VAR_CB']['+INFO'].iloc[0])] * n_trials

    # Registered values (out of loop)
    # reward_side = df[df.MSG == 'REWARD_SIDE']['+INFO'].iloc[-1]  # [-1] to take the last one in case CB was on
    # valve_1 = [float(df[df.MSG == 'VALVE_1']['+INFO'].iloc[0])] * n_trials
    # valve_2 = [float(df[df.MSG == 'VALVE_2']['+INFO'].iloc[0])] * n_trials

    # Data curation
    # creator = [creator.split()[0][2:-2]] * n_trials
    # subject = [subject.split()[0][2:-2]] * n_trials
    # date = [session_started.split()[0]] * n_trials
    # time_session_started = [session_started.split()[1]] * n_trials
    # time_session_ended = [session_ended.split()[1]] * n_trials
    reward_side = reward_side[1:-1].split(',')  # Convert string to list, [1:-1] to get rid of the square brackets []
    reward_side = list(map(int, reward_side))  # Convert list elements from string to integers
    # reward_side = np.array(reward_side, dtype=int)  # Convert to array
    reward_side = reward_side[:n_trials]

    ####################################################################################################################

    # Initialize lists
    # reward_side is already computed
    trial = []  # Useful for merging sessions later on
    rep_trial = []  # Trials in which the rewarded side coincides with the previous animal’s response (not rewarded side).
    reward = []
    punish = []
    miss = []
    wrong_lick = []
    hit = []
    after_hit = []  # After correct
    choice = []
    rep_choice = []
    response = []
    trial_start = []
    trial_end = []
    trial_len = []
    stim_start = []
    stim_end = []
    stim_len = []  # Adddddddd
    resp_win_start = []
    resp_win_end = []
    resp_win_len = []
    filename = []
    filename2 = []
    evidence = []
    evi_rep = []
    coherence = []
    # coh_rep = []
    port1in = []
    port1out = []
    port2in = []
    port2out = []

    # Test sound left/right
    sound_detec_left = []
    sound_detect_right = []
    sound_detect = []

    ####################################################################################################################

    for i in range(len(index) - 1):  # -1 to not take into account last trial

        trial.append(i)
        band = df[index[i]:index[i + 1]]

        if pd.isnull(band[(band.TYPE == 'STATE') & (band.MSG == 'Reward')]['+INFO'].iloc[0]) == False:
            reward.append(1)
        else:
            reward.append(0)

        if pd.isnull(band[(band.TYPE == 'STATE') & (band.MSG == 'Punish')]['+INFO'].iloc[0]) == False:
            punish.append(1)
        else:
            punish.append(0)

        if pd.isnull(band[(band.TYPE == 'STATE') & (band.MSG == 'Miss')]['+INFO'].iloc[0]) == False:
            miss.append(1)
            response.append(0)
        else:
            miss.append(0)
            response.append(1)

        # I prefer to avoid try-except code blocks. Workaround: if stage 1 do this else pass
        try:
            if pd.isnull(band[(band.TYPE == 'STATE') & (band.MSG == 'WrongLick')]['+INFO'].iloc[0]) == False:
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
            # Another way is check PortIns

        if wrong_lick[i] == 1:  # Consider wrong licks as punish
            hit[i] = 0

        # Trial timestamps
        trial_start.append(band[band.TYPE == 'INFO']['BPOD-INITIAL-TIME'].iloc[0])
        trial_end.append(band[band.TYPE == 'INFO']['BPOD-FINAL-TIME'].iloc[0])
        trial_len.append(float(band[band.TYPE == 'INFO']['+INFO'].iloc[0]))

        # Stimulus timestamps (take BPOD-FINAL-TIME as the real timestamp of the TTL = BPOD-INITIAL-TIME + TTL duration)
        stim_start.append(float(
            band[(band['TYPE'] == 'STATE') & (band['MSG'] == 'StimulusTrigger')]['BPOD-FINAL-TIME'].iloc[0]))

        # This if block is because the finite state machine only goes over 'StimulusStop' after a Hit
        if miss[i] == 1:
            stim_end.append(float(band[(band['TYPE'] == 'STATE') & (band['MSG'] == 'Miss')]['BPOD-FINAL-TIME'].iloc[0]))
        elif punish[i] == 1:
            stim_end.append(float(band[(band['TYPE'] == 'STATE') & (band['MSG'] == 'Punish')]['BPOD-FINAL-TIME'].iloc[0]))
        else:  # Reward or WrongLick
            stim_end.append(float(band[(band['TYPE'] == 'STATE') & (band['MSG'] == 'StimulusStop')]['BPOD-FINAL-TIME'].iloc[0]))

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

        try:
            filename2.append(band[(band['TYPE'] == 'VAL') & (band['MSG'] == 'FILENAME2')]['+INFO'].iloc[0])  # Arduino
            # sounds
        except IndexError:
            filename2.append(np.nan)

        evidence.append(float(band[(band['TYPE'] == 'VAL') & (band['MSG'] == 'EVIDENCE')]['+INFO'].iloc[0]))
        coherence.append(float(band[(band['TYPE'] == 'VAL') & (band['MSG'] == 'COHERENCE')]['+INFO'].iloc[0]))
        # Sound filename + sound2 filename + coherence/evidence + presented coherences/evidences
        # Stage, motor, substage for tracking changes within session when running a single script
        # Register running window

        # Sound left or right
        if band[(band['TYPE'] == 'EVENT') & (band['+INFO'] == 'BNC1High')].size > 0:
            sound_detec_left.append(1)
        else:
            sound_detec_left.append(0)

        if band[(band['TYPE'] == 'EVENT') & (band['+INFO'] == 'BNC2High')].size > 0:
            sound_detect_right.append(1)
        else:
            sound_detect_right.append(0)

        sound_detect.append(sound_detec_left[i] + sound_detect_right[i])

        if i == 0:
            after_hit.append(np.nan)
            rep_choice.append(np.nan)
            rep_trial.append(np.nan)
            evi_rep.append(np.nan)
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

            # RepTrial or RewardRepeat?
            # Trials in which the rewarded side coincides with the previous animal’s response (not rewarded side).
            if np.isnan(choice[i - 1]):
                rep_trial.append(np.nan)
            elif reward_side[i] == choice[i - 1]:
                rep_trial.append(1)  # Repeat trials
            else:
                rep_trial.append(0)  # Alternated trials

            # Evidence/coherence repeat
            if np.isnan(rep_trial[i]):
                evi_rep.append(np.nan)
                # coh_rep.append(np.nan)
            elif rep_trial[i] == 0:  # Negative evi/coh
                evi_rep.append(-abs(evidence[i]))
                # coh_rep.append(-abs(coherence[i]))
            elif rep_trial[i] == 1:  # Positive evi/coh
                evi_rep.append(abs(evidence[i]))
                # coh_rep.append(abs(coherence[i]))

    ####################################################################################################################

    # Construct DataFrame
    columns = ['Trial', 'Side', 'RepTrial', 'Reward', 'Punish', 'Miss', 'WrongLick', 'Hit', 'AfterHit', 'Choice',
               'RepChoice', 'Response', 'TrialStart', 'TrialEnd', 'TrialLen', 'StimStart', 'StimEnd', 'StimLen',
               'RespWinStart', 'RespWinEnd', 'RespWinLen', 'Filename', 'Filename2', 'Evidence', 'EviRep', 'Coherence',
               'Port1In', 'Port1Out', 'Port2In', 'Port2Out', 'sound_detect_left', 'sound_detect_right', 'sound_detect',
               'Stage']

    data = list(zip(trial, reward_side, rep_trial, reward, punish, miss, wrong_lick, hit, after_hit, choice,
                    rep_choice, response, trial_start, trial_end, trial_len, stim_start, stim_end, stim_len,
                    resp_win_start, resp_win_end, resp_win_len, filename, filename2, evidence, evi_rep, coherence,
                    port1in, port1out, port2in, port2out, sound_detec_left, sound_detect_right, sound_detect, stage))

    new_df = pd.DataFrame(data=data, columns=columns)

    if file.df is None:
        file.df = new_df
    else:
        trials = file.df.shape[0]
        new_df.Trial = new_df.Trial + trials
        file.df = pd.concat([file.df, new_df])

    file.skip += index[-1] - 1
