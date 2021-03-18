import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from collections import defaultdict
from collections import namedtuple

import warnings

#path = 'C:/Users/user/Google Drive (tiffany.ona@gmail.com)/WORKING_MEMORY/EXPERIMENTS/3B/setups/C18/C18_StageTraining_2B_V4_20190511-135832.csv'

path ='C:/Users/Tiffany/Google Drive/WORKING_MEMORY/EphysB/setups/E03/E03_StageTraining_Ephys_V1_20210212-122507.csv'
# LOGGING AND WARNINGS
logger = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# MAIN PARSE FUNCTION
def wm_parse(file):
    # PARSING THE DF
    try:
        df = pd.read_csv(file.path, skiprows=6, sep=';')
    except FileNotFoundError:
        logger.critical("Error: Reading CSV file. Exiting...")
        raise

    # VALUES WE HAVE ALREADY EXTRACTED FROM THE FILENAME
    session_name = file.session_name
    task = file.task
    subject_name = file.subject_name
    date = file.date
    day = file.day
    time = file.time
    #This extracts metadata from the pandas dataframe and saves it into a namedtuple.

    # subject_name = 'C18'
    # day = '1'
    # time = '1'
    # session_name = 'test'
    # date = '1'
    # task = 'test'
    
    # READS METADATA
    # Detect the box at which animal was trained
    try:
        box = df[df.MSG == 'BOARD-NAME']['+INFO'].iloc[0]
    except IndexError:
        box = "Unknown box"
        logger.warning("Box name not found.")
    
    # Stage number
    try:
        stage_number = df[df.MSG == 'STAGE_NUMBER']['+INFO'].iloc[0]
        stage_number = int(stage_number)
    except IndexError:
        stage_number = np.nan
        logger.warning("Stage number not found.")
    
    # Delay medium (in the setup is name without the M)
    try:
        delay_m = df[df.MSG == 'VAR_DELAY']['+INFO'].iloc[0]
        delay_m = float(delay_m)
        # Delay medium form when we used the cue (LEGACY)
        if delay_m < 0.1:
            delay_m = df[df.MSG == 'VAR_CUE']['+INFO'].iloc[0]
            delay_m = float(delay_m)
    except IndexError:
        delay_m = np.nan
        logger.warning("Delay duration not found.")

    # Delay long (in the setup is names delay_h)
    try:
        delay_h = df[df.MSG == 'VAR_DELAY_H']['+INFO'].iloc[0]
        delay_h = float(delay_h)
    except IndexError:
        delay_h = np.nan
        logger.warning("Delay high duration not found.")

    # Delay short (in the setup is names delay_l)
    try:
        delay_l = df[df.MSG == 'VAR_DELAY_L']['+INFO'].iloc[0]
        delay_l = float(delay_l)
    except IndexError:
        delay_l = np.nan
        logger.warning("Delay low duration not found.")

    # Duration of fixation
    try:
        fixation = df[df.MSG == 'VAR_FIXATION']['+INFO'].iloc[0]
        fixation = float(fixation)
    except IndexError:
        fixation = np.nan
        logger.warning("Fixation not found.")

    # Whether there was timeout or not. 
    try:
        timeout = df[df.MSG == 'VAR_TIMEOUT']['+INFO'].iloc[0]
        timeout = float(timeout)
    except IndexError:
        timeout = np.nan
        logger.warning("Timeout not found.")
        
    try:
        lick = df[df.MSG == 'VAR_TIMEOUT']['+INFO'].iloc[0]
        lick = float(lick)
    except IndexError:
        lick = np.nan
        logger.warning("Lick not found.")

    try:
        substage = df[df.MSG == 'VAR_SUBSTAGE']['+INFO'].iloc[0]
        substage = float(substage)
    except IndexError:
        substage = np.nan
        logger.warning("Substage not found.")

    try:
        motor = df[df.MSG == 'VAR_MOTOR']['+INFO'].iloc[0]
        motor = float(motor)
    except IndexError:
        motor = np.nan
        logger.warning("Substage not found.")

    try:
        delay_progression_value = df[df.MSG == 'VAR_DELAY_PROGRESSION']['+INFO'].iloc[0]
        delay_progression_value = float(delay_progression_value)
    except IndexError:
        delay_progression_value = np.nan
        logger.warning("Delay progression not found.")

    try:
        switch = df[df.MSG == 'VAR_SWITCH']['+INFO'].iloc[0]
        switch = float(switch)
    except IndexError:
        switch = np.nan
        logger.warning("Delay progression not found.")
        
    try:
        catch = df[df.MSG == 'VAR_CATCH']['+INFO'].iloc[0]
        catch = float(catch)
    except IndexError:
        catch = np.nan
        logger.warning("Delay progression not found.")

    try:
        drug = df[df.MSG == 'VAR_DRUG']['+INFO'].iloc[0]
        drug = str(drug)
    except IndexError:
        drug = 0
        logger.warning("No drug")

    logger.info(str(subject_name) + str(day) + '-' + str(time))
    logger.info("Session metadata loaded.")

    #    Extracts the main vectors and data from the CSV file: reward_side, hithistory, session length, response time, coherences and transition information (for the poke histogram).
    # -----------------------------------------------------------------------------
    # Extract the main states of the session. They contain a value if the trial passed
    # through that state, NaN otherwise. Here, the main states are punish,
    # wronglick, reward, fixation break and miss.

    punish_data = df.query("TYPE=='STATE' and MSG=='Punish'")['BPOD-FINAL-TIME'].astype(float)
    reward_data = df.query("TYPE=='STATE' and MSG=='Reward'")['BPOD-FINAL-TIME'].astype(float)
    misses = df.query("TYPE=='STATE' and MSG=='Miss'")['BPOD-FINAL-TIME'].astype(float)
    AW = df.query("TYPE=='STATE' and MSG=='AW'")['BPOD-FINAL-TIME'].astype(float)
    iti = df.query("TYPE=='STATE' and MSG=='ITI'")['BPOD-FINAL-TIME'].astype(float)
    wronglick = df.query("TYPE=='STATE' and MSG=='WrongLick'")['BPOD-FINAL-TIME'].astype(float)

    # Since the state machines are designed so that the main states are mutually exclusive in a single trial, we
    # can compute the total session length as the sum of all the values that are not NaN:

    #        length = (punish_data.dropna().size
    #                        + reward_data.dropna().size
    #                        + misses.dropna().size)

    # NUMBER OF TRIALS
    trial_end = np.flatnonzero(df['TYPE'] == 'END-TRIAL')
    length = int(len(trial_end))  # only ended trials
    trials = range(length)

    #        trial_num = len(df.query("TYPE=='TRIAL' and MSG=='New trial'"))
    #        length = trial_num
    #        print(trial_num)

    # Detect which are the indices of those states that make a trial not be valid
    # (here, miss and invalid):
    miss_indices = np.where(~np.isnan(misses))[0]
    total_invalid_indices = miss_indices

    # Wrong lick trials
    WL_indices = np.where(~np.isnan(wronglick))[0]

    # Compute hithistory vector; it will now contain True if the answer
    # was correct, False otherwise. It does the same for the misses.
    hithistory = np.logical_not(np.isnan(reward_data.values)[:length])
    misshistory = np.logical_not(np.isnan(misses.values)[:length])
    validhistory = np.logical_not(~np.isnan(misses.values)[:length])
    AWhistory = np.logical_not(np.isnan(AW.values)[:length])
    wronglickhistory = hithistory.copy()

    try:
        for i in WL_indices[:length]:
            wronglickhistory[i] = False
            
    except IndexError:
        pass

    total_trials = length
    correct_trials = sum([h == 1 for h in hithistory])
    invalid_trials = sum(misshistory)
    valid_trials = total_trials - invalid_trials

    prob_motor_stage = np.repeat(np.nan, length)
    motor_stage = np.repeat(np.nan, length)

    prob_motor_stage = df[df.MSG == "PROB MOTOR STAGE"]['+INFO'].values[:length].astype(float)
    motor_stage = df[df.MSG == "MOTOR STAGE"]['+INFO'].values[:length].astype(float) 
    
    # --------------------------------------------------------------------------------
    # REWARD_SIDE contains a 1 if the correct answer was the (R)ight side, 0 otherwise.
    # It is received as a string from the CSV: "[0,1,0,1,0,1,1,1,1,0,0,1,0,1,1,1]"...
    # and there can be more than one in the file. We always take THE LAST (iloc[-1]).
    try:
        reward_side_str = df[df.MSG == "REWARD_SIDE"]['+INFO'].iloc[-1][1:-1].split(',')[:length]
    except IndexError:  # Compatibility with old files, when REWARD_SIDE was called VECTOR_CHOICE.
        logger.warning("REWARD_SIDE vector not found. Trying old VECTOR_CHOICE...")
        try:
            reward_side_str = df[df.MSG == 'VECTOR_CHOICE']['+INFO'].iloc[-1][1:-1].split(',')[:length]
        except IndexError:
            raise TypeError("Neither REWARD_SIDE nor VECTOR_CHOICE found. Exiting...")
    else:
        # Cast to int from str:
        reward_side = np.array(reward_side_str, dtype=int)
    
    # Variable that tells in which block we are in
    prob_repeat = df[df.MSG == 'prob_repeat']['+INFO'].values[:length].astype(float)

    # coherences vector, from 0 to 1 (later it will be converted
    # into evidences from -1 to 1):
    coherences = df[df.MSG == 'DB']['+INFO'].values[:length].astype(float)

    # Delete invalid trials:
    coherences_valids = np.delete(coherences, total_invalid_indices)

    coherences_left = df[df.MSG == 'LEFT_COH']['+INFO'].values[:length].astype(float)
    coherences_right = df[df.MSG == 'RIGHT_COH']['+INFO'].values[:length].astype(float)

    if not coherences.size:
        logger.warning("This trial doesn't use coherences.")
        coherences = np.repeat(np.nan, length)
        coherences_left = np.repeat(np.nan, length)
        coherences_right = np.repeat(np.nan, length)

    presented_coherences = 0
    try:
        presented_coherences = np.unique(coherences)
    except:
        pass
    
    # Probabilities for delay -----------------------------------------------------------------------------
    if stage_number >= 3:
        probabilities = df[df.MSG == 'PROB']['+INFO'].values[:length].astype(str)
    else:
        probabilities = np.repeat(np.nan, length)

    # ------------------- Delay measurements -------------------------------------------
    if motor == 6:
        # If the session has delay progression activated, it takes the delay value from the register option as well asthe delay progression value.
#        if delay_progression_value == 1:
##            delay_progression = df.query("TYPE=='VAL' and MSG=='DELAY_PROGRESSION'")['+INFO'].values[:length].astype(float)
##            delay_progression = np.concatenate((np.repeat(delay_l, 20), delay_progression))
##        else:
#            delay_progression = np.repeat(delay_progression_value, length)
#        else:
        delay_progression = np.repeat(delay_progression_value, length)
        
        try:
            delay_times = df.query("TYPE=='VAL' and MSG=='DELAY'")['+INFO'].values[:length].astype(float)
        except:
            delay_times = np.repeat(np.nan, length)
            pass
        if stage_number >= 3:
            # Calculate the values of high delay and add the initial 30 trials with proportion
            delay_times_h = df[df.MSG == 'DELAY_PROGRESSION_H']['+INFO'].values[:length].astype(float)

            # Calculate the values of medium delay and add the initial 30 trials with proportion
            delay_times_m = df[df.MSG == 'DELAY_PROGRESSION_M']['+INFO'].values[:length].astype(float)
            delay_times_l = np.repeat(delay_l, length)
        else:
            delay_times_h = np.repeat(np.nan, length)
            delay_times_m = np.repeat(np.nan, length)
            delay_times_l = np.repeat(np.nan, length)            

        
        if len(trials) != len(delay_times):
            delay_times = np.append(delay_times, 0.1)

        if len(delay_times) != len(delay_times_h):
            delay_times_h = np.insert(delay_times_h,0,np.repeat(delay_h,len(delay_times)-len(delay_times_h)))
            
        if len(delay_times) != len(delay_times_m):
            delay_times_m = np.insert(delay_times_m,0,np.repeat(delay_m,len(delay_times)-len(delay_times_m)))           
 
        delay_types = []            
        for trial in trials:
            if delay_times[trial] == round(delay_times_l[trial], 3):
                delay_types.append('delay_l')

            elif delay_times[trial] == round(delay_times_m[trial], 3):
                delay_types.append('delay_m')

            elif delay_times[trial] == delay_times_h[trial]:
                delay_types.append('delay_h')

            else:
                delay_types.append('delay_h')
               
    else:
        delay_progression = np.repeat(np.nan, length)
        delay_times = np.repeat(np.nan, length)
        delay_times_h = np.repeat(np.nan, length)
        delay_times_m = np.repeat(np.nan, length)
        delay_times_l = np.repeat(np.nan, length)
        delay_types =np.repeat(np.nan, length)

    # Measure which are the delays that are presented. This is used for those sessions with different delay lengths such as 2 ,5 or 10
    presented_delays = 0
    if delay_progression_value == 0 and stage_number == 3:
        presented_delays = np.unique(delay_times)
        
    # Prepare columns for further measurements. 
    repeat_delay = np.repeat(np.nan,length)      
    after_correct = np.repeat(np.nan,length)
    repeat_side = np.repeat(np.nan,length)
    # repeat_choice = np.repeat(np.nan,length)
    
    for i in range(length):
        if i != 0: #All the following analysis can't be done in the first trial
            if hithistory[i-1] == True:
                after_correct[i]=1 # After correct
            elif  hithistory[i-1] == False and validhistory[i] == False:
                after_correct[i]=2 # After miss
            elif  hithistory[i-1] == False and validhistory[i] == True:
                after_correct[i]=0 # After incorrect

            if reward_side[i-1] == reward_side[i]:
                repeat_side[i] = 1
            else:
                repeat_side[i] = 0

            # if reward_side[i-1] == reward_side[i] and hit[i]==True and hit[i-1]==True or hit[i]==False and hit[i-1]==False:
            #     repeat_choice[i] = 1
            # else:
            #     repeat_choice[i] = 0               
                
            if delay_types[i-1] ==  delay_types[i]:
               repeat_delay[i] = 1
            elif delay_types[i-1] !=  delay_types[i]:
               repeat_delay[i] = 0
                
    session_params = {'session_name': session_name,
                      'stage_number': stage_number,
                      'subject_name': subject_name,
                      'day': day,
                      'time': time,
                      'box': box,
                      'date': date,
                      'task': task,
                      'stage_number': stage_number,
                      'fixation': fixation,
                      'timeout': timeout,
                      'lick': lick,
                      'motor': motor,
                      'substage': substage,
                      'switch': switch,
                      'catch': catch,
                      'total_trials': total_trials,
                      'valid_trials': valid_trials,
                      'correct_trials': correct_trials,
                      'invalid_trials': invalid_trials,
                      'delay_progression_value': delay_progression_value,
                      'delay_l': delay_l,
                      'delay_m': delay_m,
                      'delay_h': delay_h,
                      'presented_delays': presented_delays,
                      'presented_coherences': presented_coherences,
                      'drug':drug,
                      }

    logger.info("Session raw data loaded.")

    session_trials = {'trials': np.arange(length),
                      'reward_side': reward_side,
                      'wronglickhistory': wronglickhistory,
                      'hithistory': hithistory,
                      'misshistory': misshistory,
                      'AWhistory': AWhistory,
                      'probabilities': probabilities,
                      'validhistory': validhistory,
                      'delay_times': delay_times,
                      'delay_types': delay_types,
                      'delay_progression': delay_progression,
                      'delay_times_l': delay_times_l,
                      'delay_times_m': delay_times_m,
                      'delay_times_h': delay_times_h,
                      'coherences': coherences,
                      'coherences_right': coherences_right,
                      'coherences_left': coherences_left,
                      'after_correct': after_correct,
                      'repeat_delay': repeat_delay,
                      'repeat_side': repeat_side,
                      'motor_stage':motor_stage,
                      'prob_motor_stage': prob_motor_stage,
                      'prob_repeat': prob_repeat
                      }
					  
    parsed = parsed_events(df, length)
    session_trials.update(parsed[0])
    session_trials.update(parsed[1])
    session_params.update(parsed[2])

    return [session_params, session_trials]

def parsed_events(df, length):
    """
    Extracts information about the session's state and poke timestamps. Works with any session
    type. state_list contains a list of all the unique states; state_timestamps contains the
    timestamps for each state, both _start and _end.
    """
    # --------------- Computation of the states list ----------
    new_trial = df.query("TYPE=='TRIAL' and MSG=='New trial'").index[1]
    last_trial = df.query("TYPE=='END-TRIAL'").index[0]
    trial_band_states = df.query("TYPE=='STATE' and index > @last_trial and index < @new_trial")
    state_list = set(trial_band_states['MSG'].values)

    states = defaultdict(list)
    pokes = defaultdict(list)

    new_trial_indexes = df.query("TYPE=='TRIAL' and MSG=='New trial'").index[:length]
    for jj in range(length):

        if jj != length - 1:
            trial_band = df[new_trial_indexes[jj]:new_trial_indexes[jj + 1]]
        else:
            trial_band = df[new_trial_indexes[jj]:]
        trial_band_events = trial_band.query("TYPE=='EVENT'")
        trial_band_states = trial_band.query("TYPE=='STATE'")
        # -------------------------- STATE TIMESTAMPS ----------------------------------- #
        for state in state_list:
            start_times = trial_band_states.query("MSG==@state")['BPOD-INITIAL-TIME'].values.tolist()
            end_times = trial_band_states.query("MSG==@state")['BPOD-FINAL-TIME'].values.tolist()
            states[str(state) + '_start'].append(start_times)
            states[str(state) + '_end'].append(end_times)

        # --------------------------- POKE TIMESTAMPS ----------------------------------- #
        found_L_in = found_C_in = found_R_in = found_L_out = found_C_out = found_R_out = False
        C_start = []
        L_start = []
        R_start = []
        C_end = []
        L_end = []
        R_end = []
        for (_, row) in trial_band_events.iterrows():
            if row['+INFO'] == 'Port1In':  # Port1In
                found_L_in = True
                if found_L_out:
                    L_start.append(np.nan)
                    found_L_out = False
                L_start.append(row['BPOD-INITIAL-TIME'])
            elif row['+INFO'] == 'Port2In':  # Port2In
                found_C_in = True
                if found_C_out:
                    C_start.append(np.nan)
                    found_C_out = False
                C_start.append(row['BPOD-INITIAL-TIME'])
            #                elif row['+INFO'] == 'Port3In': # Port3In
            #                    found_R_in = True
            #                    if found_R_out:
            #                        R_start.append(np.nan)
            #                        found_R_out = False
            # R_start.append(row['BPOD-INITIAL-TIME'])
            elif row['+INFO'] == 'Port1Out':  # Port1Out
                if not found_L_in:
                    found_L_out = True
                else:
                    found_L_in = False
                L_end.append(row['BPOD-INITIAL-TIME'])
            elif row['+INFO'] == 'Port2Out':  # Port2Out
                if not found_C_in:
                    found_C_out = True
                else:
                    found_C_in = False
                C_end.append(row['BPOD-INITIAL-TIME'])
            #                elif row['+INFO'] == 'Port3Out': # Port3Out
            #                    if not found_R_in:
            #                        found_R_out = True
            #                    else:
            #                        found_R_in = False
            #                    R_end.append(row['BPOD-INITIAL-TIME'])
            else:
                pass

        if found_L_in:
            L_end.append(np.nan)
        if found_C_in:
            C_end.append(np.nan)
        if found_R_in:
            R_end.append(np.nan)

        if found_L_out:
            L_start.append(np.nan)
        if found_C_out:
            C_start.append(np.nan)
        if found_R_out:
            R_start.append(np.nan)

        pokes['L_e'].append(L_end)
        pokes['C_e'].append(C_end)
        pokes['C_s'].append(C_start)
        pokes['L_s'].append(L_start)

    statelist = {'state_list': list(state_list)}

    return [states, pokes, statelist]
