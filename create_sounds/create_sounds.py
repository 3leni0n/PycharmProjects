# Import libraries
import time
import numpy as np
import sys
import os
import itertools
import wavio
from pydub import AudioSegment
import pandas as pd
from matplotlib import pyplot as plt
import string
from my_fun.my_fun import *

########################################################################################################################

def create_sounds_v1(save=False):
    """Function to create the sounds set for an ILD 2AFC task. A white noise vector will be generated, and then its
    amplitude will fluctuate through an envelope to produce sounds with a given evidence. Since the task consist in
    determining from what side, left or right, the sound is louder on average, the evidence represent the information
    available to make a choice for each sound, being -1 left speaker only and +1 right speaker only. The total number of
    sounds produced is n evidences ** 2. The filename is a combination of 3 lowercase letters. The function saves the
    files and a table with an envelope value per frame per side in amplitude.
    save: To save or not the sound files and its corresponding csv. False by default to avoid overwriting the current
    files by mistake and for performance reasons when calling the function to simulate sound sets.
    """

    time_start = time.time()

    # Generate white noise
    noise = white_noise(fs=44100, cutoff=[2000, 20000], amp=1, dur=1, fn=10000, normalize=False)
    # band_fs=[2000, 20000] as in rat's tasks. Human range is 20-20000 and mice 1000-70000
    # FsOut=44100 the most used (audio CD)

    ####################################################################################################################

    # Generate evidences and coherences
    evidences = np.array([-1, -0.9, -0.8, -0.75, -0.6, -0.5, -0.4, -0.3, -0.25, -0.1,
                          0, 0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1])

    # (max_vol/min_value)^(1/n-1)

    # Evidences spaced 0.1 because len(np.arange(-1, 1, 0.1)) < len(list(string.ascii_lowercase)). In words, the sounds'
    # filename can be expressed only with letters (20 characters), while an spacing of 0.05 would require 40 characters
    # and use numbers too
    # Evidences finished in .05 for the final task and psychometric curves
    coherences = evi2coh(evidences)  # From 0 (left) to 1 (right) so 0 net evidence returns 0.5. Input argument needed
    # to calculate beta distribution in envelope function

    ####################################################################################################################

    # Select the folder and create it if it doesn't exist
    # folder = '/home/alexis/Música/sounds/'
    folder = '/home/alexis/Escritorio/test/'

    if not os.path.exists(folder):
        os.mkdir(folder)

    # 21 * 21 chars = 441 possible sounds per evidence
    chars = list(string.ascii_lowercase[:len(evidences)])  # Make a list of all the lowercase letters as long as
    # evidences

    # Create DataFrame column labels
    columns = ['filename',
               'EL0', 'EL1', 'EL2', 'EL3', 'EL4', 'EL5', 'EL6', 'EL7', 'EL8', 'EL9',  # Envelope Left * 10 frames
               'ER0', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6', 'ER7', 'ER8', 'ER9']  # Envelope Right * 10 frames

    # [f'EL{n:02}' for n in range(n_frames)]  # To iterate

    sound_number = 0  # Initialize counter
    ELER = []  # Envelope left + envelope right

    for k in range(len(chars)):

        for i, j in itertools.product(chars, chars):  # Iterate through all the possible combinations of chars
            # Sound number (name) from 1 (aaa) to 9261 (uuu)
            sound_number += 1
            name = folder + chars[k] + i + j

            filename = chars[k] + i + j  # For the csv file

            path_wav = name + '.wav'
            # path_mp3 = name + '.mp3'
            # path_ogg = name + '.ogg'
            print(sound_number, name)

            SL, SR, EL, ER = envelope(noise, coherences[k], fs=44100, amp=1, dur=1, n_frames=10, var=0.015,
                                      paired=False)

            ELER.append([filename] + list(EL) + list(ER))

            if save == True:  # Save sounds only if specified (don't wanna for simulation purposes)
                sound = np.column_stack((filename, SL, SR))
                wavio.write(path_wav, sound, 44100, sampwidth=1)  # Write the array sound to a wav file
                # sound_wav = AudioSegment.from_wav(path_wav)  # Read the wav file to a wav sound
                # sound_wav.export(path_mp3, format='mp3')  # Export the wav sound to a mp3 file
                # sound_wav.export(path_ogg, format='ogg')  # Export the wav sound to a ogg file

    df = pd.DataFrame(data=ELER, index=None, columns=columns)

    if save == True:
        # df_ild.to_csv('sounds.csv')  # Save df_ild as csv file
        df.to_csv(folder + 'test.csv')

    time_end = time.time()
    runtime = time_end - time_start
    print('\nThe script took', round(runtime, 2), 'seconds to run')

    return df
