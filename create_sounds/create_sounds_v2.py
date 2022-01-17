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

max_vol = 70  # Calibration value
ILDs_dB = np.array([-70, -10, -4, -2, 0, 2, 4, 10, 70])

dBs = []
for i in ILDs_dB:
    value = get_dBs_from_diff(i, max_vol)
    dBs.append(value)

dBs = np.round(dBs)
dBs_right = list(abs(np.unique(dBs.flatten())))
dBs_left = list(np.flip(dBs_right))

sigma = 1


def create_sounds_v2(save=False):
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
    noise = white_noise(fs=44100, cutoff=[2000, 20000], amp=1, dur=1, fn=10000, normalize=True)
    # band_fs=[2000, 20000] as in rat's tasks. Human range is 20-20000 and mice 1000-70000
    # FsOut=44100 the most used (audio CD)

    ####################################################################################################################

    # Select the folder and create it if it doesn't exist
    # folder = '/home/alexis/Música/sounds/'
    folder = '/home/alexis/Escritorio/test/'

    if not os.path.exists(folder):
        os.mkdir(folder)

    # 21 * 21 chars = 441 possible sounds per evidence
    chars = list(string.ascii_lowercase)  # Make a list of all the lowercase letters as long as
    # evidences

    # Create DataFrame column labels
    columns = ['filename',
               'EL0', 'EL1', 'EL2', 'EL3', 'EL4', 'EL5', 'EL6', 'EL7', 'EL8', 'EL9',  # Envelope Left * 10 frames
               'ER0', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6', 'ER7', 'ER8', 'ER9']  # Envelope Right * 10 frames

    # [f'EL{n:02}' for n in range(n_frames)]  # To iterate

    sound_number = 0  # Initialize counter
    ELER = []  # Envelope left + envelope right

    for k in range(len(ILDs_dB)):

        dB_left = dBs_left[k]
        dB_right = dBs_right[k]

        for i, j in itertools.product(chars, chars):  # Iterate through all the possible combinations of chars
            # Sound number (name) from 1 (aaa) to 9261 (uuu)
            sound_number += 1
            name = folder + chars[k] + i + j

            filename = chars[k] + i + j  # For the csv file

            path_wav = name + '.wav'
            # path_mp3 = name + '.mp3'
            # path_ogg = name + '.ogg'
            print(sound_number, name)

            SL, SR, EL, ER = do_envelope_dB_normal(noise, dB_left, dB_right, max_vol,
                                                   fs=44100, amp=1, dur=1, n_frames=10, sigma=sigma)

            ELER.append([filename] + list(EL) + list(ER))

            if save == True:  # Save sounds only if specified (don't wanna for simulation purposes)
                sound = np.column_stack((filename, SL, SR))
                wavio.write(path_wav, sound, 44100, sampwidth=1)  # Write the array sound to a wav file
                # sound_wav = AudioSegment.from_wav(path_wav)  # Read the wav file to a wav sound
                # sound_wav.export(path_mp3, format='mp3')  # Export the wav sound to a mp3 file
                # sound_wav.export(path_ogg, format='ogg')  # Export the wav sound to a ogg file

            if k == 0 or k == 8:
                break  # Not to create more than 1 sound for maximum evidence

    df = pd.DataFrame(data=ELER, index=None, columns=columns)

    if save == True:
        # df_ild.to_csv('sounds_1.csv')  # Save df_ild as csv file
        df.to_csv(folder + 'test.csv')

    time_end = time.time()
    runtime = time_end - time_start
    print('\nThe script took', round(runtime, 2), 'seconds to run')

    return df
