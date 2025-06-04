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

########################################################################################################################

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
df = pd.read_csv(path_sounds / 'sounds.csv')
dict_sounds = make_sounds_dict(0.0003, 0.0078, 26, 4)  # Change name of variable as it's the same of the function
ilds = df.ILD.unique()

########################################################################################################################

# Task variables
n_trials = 1000  # Number of trials
trial_types = [0, 1]  # 0 (sound left) or 1 (sound right)
p = 1  # Probability of non-maximum evidence trials (i.e. difficulty)
p_right = 0.5  # Probability of right side trials
trial_list = np.random.choice(trial_types, n_trials, p=[1 - float(p_right), float(p_right)]).tolist()  # Trials dist
resp_win = 1  # Response window in seconds
iti = 3  # Inter-trial interval in seconds
stim_dur = 1.1  # Stimulus duration in seconds
delay = 0.5  # Delay between stimulus and response in seconds
feedback = ['Correct', 'Error', 'Miss']  # Feedback messages

########################################################################################################################

# Initialize counters
session_start_time = start_time = time.time()
responses = 0
misses = 0
hits = 0
errors = 0

# Initialize trial data storage
trial_ild = []

########################################################################################################################

# Main loop

for trial in range(n_trials):

    trial_start_time = time.time()  # Start time of the trial
    print(f'Trial: {trial}')
    side = trial_list[trial]  # Get the side for this trial
    ild = select_ilds(ilds, p, side)  # Select ILD based on the side and probability
    trial_ild.append(ild)  # Store the ILD for this trial

    # Play sound
    sample_index = df[df.ILD == ild].index  # Get the index of the sound samples corresponding to the ILD
    trial_sound = np.random.choice(sample_index)
    print(f'Sound {trial_sound}: {df.filename[trial_sound]} ({ild} dB)')
    filename = df.filename[trial_sound]
    path_sound = Path(path_sounds / filename).with_suffix('.wav')  # Path to the sound file
    data, fs = sf.read(path_sound)  # Load sound
    sd.play(data, samplerate=fs)  # Play sound
    sd.wait()  # Wait until sound finishes playing

    # Wait for response
    # Log user response with keyboard stroke
    # Record reaction time
    print("Press left or right arrow key...")
    response_start = time.time()

    while True:
        if keyboard.is_pressed('left'):
            response = 'left'
            break
        elif keyboard.is_pressed('right'):
            response = 'right'
            break
        elif time.time() - response_start > resp_win:  # Timeout
            response = 'miss'
            break

    rt = time.time() - response_start

    if response == 'miss':
        print("No response (miss)")
        misses += 1
    else:
        print(f"Response: {response} (RT = {rt:.3f} s)")
        responses += 1
        correct = (response == 'left' and side == 0) or (response == 'right' and side == 1)
        if correct:
            print("Feedback:", feedback[0])
            hits += 1
        else:
            print("Feedback:", feedback[1])
            errors += 1

    time.sleep(iti)  # Inter-trial interval







# Plot histogram of trial ilds to check distribution matches p
plt.figure()
counts = np.array([np.sum(trial_ild == ild) for ild in ilds])  # Count each ILD
plt.bar(ilds, counts, align='center')




