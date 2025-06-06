from string import ascii_lowercase
import numpy as np
from pathlib import Path
import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
import keyboard
from datetime import datetime

########################################################################################################################

# Define functions
def make_sounds_dict(start, stop, num, n_decimals):
    """Dictionary letter: TTL pulses. Need to be in line with Arduino's code"""
    if num > 26:
        raise ValueError("'num' cannot be higher than abc's length (26)")
    chars = list(ascii_lowercase[:num])  # Make a list of all the lowercase letters as long as num
    pulses = np.around(np.linspace(start, stop, num), n_decimals)  # Make evenly spaced TTL pulses rounded to round2
    return dict(zip(chars, pulses))


def select_ilds(ilds, p, side):
    # r = random.random()  # Generate random float between 0 and 1. PyBpod missing random library
    r = np.random.random(1)[0]  # Generate random float between 0 and 1
    if r > p:
        if side == 0:  # Left
            return ilds.min()  # Sound left only
        else:  # Right
            return ilds.max()  # Sound right only
    else:
        if side == 0:  # Left
            options = ilds[ilds <= 0]  # Left ILDs
            options = np.repeat(options, 2)  # Repeat each element of the vector
            options = options[:-1]  # Exclude one of the 0s from the vector, so it has 1/2 p than the rest
        else:  # Right
            options = ilds[ilds >= 0]  # Right ILDs
            options = np.repeat(options, 2)  # Repeat each element of the vector
            options = options[1:]  # Exclude one of the 0s from the vector, so it has 1/2 p than the rest
    # selected_ild = random.choice(options)
    selected_ild = np.random.choice(options)
    return selected_ild


########################################################################################################################

# Import sounds and create TTL-letters dictionary
path_sounds = Path.home() / 'Music' / 'sounds_first_frame_0_ILD=False'
df_sounds = pd.read_csv(path_sounds / 'sounds.csv')
dict_sounds = make_sounds_dict(0.0003, 0.0078, 26, 4)  # Change name of variable as it's the same of the function
ilds = df_sounds.ILD.unique()

########################################################################################################################

# Task variables
n_trials = 1000  # Number of trials
trial_types = [0, 1]  # 0 (sound left) or 1 (sound right)
trial_types_str = ['Left', 'Right']  # String representation of trial types
p = 1  # Probability of non-maximum evidence trials (i.e. difficulty)
p_right = 0.5  # Probability of right side trials
trial_list = np.random.choice(trial_types, n_trials, p=[1 - float(p_right), float(p_right)]).tolist()  # Trials dist
resp_win = 3  # Response window in seconds
iti = 3  # Inter-trial interval in seconds
stim_len = 1  # Stimulus duration in seconds
delay = 1  # Delay between stimulus and response in seconds
delays = np.random.uniform(0, 1, size=n_trials)
feedback = ['Correct', 'Error', 'Miss']  # Feedback messages
feedback_colors = ['\033[1;32m',  # Bold green
                   '\033[1;31m',  # Bold red
                   '\033[1;37m']  # Bold white
reset = '\033[0m'     # Reset to default
target_n_trials = 100

########################################################################################################################

# Initialize counters
responses = 0
misses = 0
hits = 0
errors = 0

# Initialize trial data storage
trial_data = []

########################################################################################################################

# Metadata

# User input
subject = input('Enter name:')
sex = input('Enter sex (M/F):')
age = input('Enter age:')

datetime = datetime.now()
session_start = time.time()

########################################################################################################################

# Intructions

print('\n')
print('Welcome to the sound amplitude discrimination task! :)')
print('You will hear stereo sounds with different volumes in the left and right channels. You need to decide in '
      'which side the volume was LOUDER')
print('Press \u2190 to choose LEFT and \u2192 to choose RIGHT. You can only choose after the CUE')
print('Press ESC to exit at any time :(\n')

print('Press ENTER to start the task...\n')
while True:
    if keyboard.is_pressed('enter'):
        break

########################################################################################################################

# Main loop

green = "\033[1;32m"  # Bold Green
red = "\033[1;31m"    # Bold Red
reset = "\033[0m"     # Reset to default

for trial in range(n_trials):

    # ESC kill switch only at the start of trial
    if keyboard.is_pressed('esc'):
        print('ESC pressed. Exiting task')
        session_end_time = time.time()  # Register session end time
        break

    trial_start = time.time()  # Start time of the trial
    print(f'Trial: {trial}')

    # Play sound
    print('Listen sound fully...')
    side = trial_list[trial]  # Get the side for this trial
    ild = select_ilds(ilds, p, side)  # Select ILD based on the side and probability
    sample_index = df_sounds[df_sounds.ILD == ild].index  # Get the index of the sound samples corresponding to the ILD
    trial_sound = np.random.choice(sample_index)
    # print(f'Sound {trial_sound}: {df.filename[trial_sound]} ({ild} dB)')
    filename = df_sounds.filename[trial_sound]
    path_sound = Path(path_sounds / filename).with_suffix('.wav')  # Path to the sound file
    sound, fs = sf.read(path_sound)  # Load sound
    stim_start = time.time()  # Start time of the stimulus
    sd.play(sound, samplerate=fs)  # Play sound
    sd.wait()  # Wait until sound finishes playing
    stim_end = time.time()  # End time of the stimulus
    stim_dur = stim_end - stim_start  # Duration of the stimulus

    time.sleep(delays[trial])  # Delay before response window starts

    # Response window
    # Register subject response with keyboard press and record reaction time (RT)
    print("Press \u2190 or \u2192 key to choose...")
    response_start = time.time()

    while True:
        if keyboard.is_pressed('left'):
            choice = 0
            break
        elif keyboard.is_pressed('right'):
            choice = 1
            break
        elif time.time() - response_start > resp_win:  # Timeout
            choice = np.nan  # Miss
            break

    response_end = time.time()
    rt = response_end - response_start

    # Feedback and outcome
    if np.isnan(choice):
        print(f'{feedback_colors[2]}{feedback[2]}{reset}')
        outcome = np.nan
        misses += 1
    else:
        print(f'{trial_types_str[choice]} choice')
        responses += 1
        if choice == side:  # Correct
            print(f'{feedback_colors[0]}{feedback[0]}{reset}')
            outcome = 1
            hits += 1
        else:  # Error
            print(f'{feedback_colors[1]}{feedback[1]}{reset}')
            outcome = 0
            errors += 1

    time.sleep(iti)  # Inter-trial interval

    trial_end = time.time()  # End time of the trial
    trial_dur = trial_end - trial_start  # Duration of the trial

    # Store trial data
    trial_data.append({
        'trial': trial,
        'side': side,
        'ild': ild,
        'filename': filename,
        'choice': choice,
        'rt': rt if not np.isnan(choice) else np.nan,
        'outcome': outcome,
        'trial_start': trial_start,
        'trial_end': trial_end,
        'trial_duration': trial_dur,
        'stim_start': stim_start,
        'stim_end': stim_end,
        'stim_duration': stim_dur,
        'session_start': session_start
    })

    if responses == target_n_trials:
        break

    print('\n')  # New line for better readability

# Convert trial data to DataFrame
df_trials = pd.DataFrame(trial_data)

# Register session end time
session_end = time.time()
df_trials['session_start'] = session_start
df_trials['session_end'] = session_end
df_trials['subject'] = subject
df_trials['sex'] = sex
df_trials['age'] = age
df_trials['datetime'] = datetime

# Save trial data to CSV
path_output = (Path.home() / 'Documents')
filename = subject + '_' + datetime.strftime('%Y-%m-%d_%H-%M-%S')
df_trials.to_csv(Path(path_output / filename).with_suffix('.csv'), index=False)

# # Plot histogram of trial ilds to check distribution matches p
# plt.figure()
# counts = np.array([np.sum(trial_ild == ild) for ild in ilds])  # Count each ILD
# plt.bar(ilds, counts, align='center')
