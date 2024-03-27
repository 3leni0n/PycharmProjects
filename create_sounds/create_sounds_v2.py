# Import libraries
import time
import numpy as np
import sys
from pathlib import Path
import os
import itertools
import wavio
from pydub import AudioSegment
import pandas as pd
from matplotlib import pyplot as plt
import string
from my_fun.my_fun import *

########################################################################################################################


def create_sounds_v2(max_vol=60, fs=44100, cutoff=[2000, 20000], amp=1, dur=1, fn=10000, normalize=True, n_frames=10,
                     sigma=1, save=False):
    """Function to create the sounds set for an ILD 2AFC task. A white noise vector will be generated, and then its
    amplitude will fluctuate through an envelope to produce sounds with a given evidence. Since the task consist in
    determining from what side, left or right, the sound is louder on average, the evidence represent the information
    available to make a choice for each sound, being -1 left speaker only and +1 right speaker only. The total number of
    sounds produced is n evidences ** 2. The filename is a combination of 3 lowercase letters. The function saves the
    files and a table with an envelope value per frame per side in amplitude.
    save: To save or not the sound files and its corresponding csv. False by default to avoid overwriting the current
    files by mistake and for performance reasons when calling the function to simulate sound sets.
    UPDATE with v2!!!!
    """

    time_start = time.time()

    ILDs_dB = np.array([-max_vol, -8, -4, -2, 0, 2, 4, 8, max_vol])

    dBs = []
    for i in ILDs_dB:
        value = get_dBs_from_diff(i, max_vol)
        dBs.append(value)

    dBs = np.round(dBs)
    dBs_right = list(abs(np.unique(dBs.flatten())))
    dBs_left = list(np.flip(dBs_right))

    ####################################################################################################################

    # Generate white noise
    noise = white_noise(fs=fs, cutoff=cutoff, amp=amp, dur=dur, fn=fn, normalize=normalize)
    # band_fs=[2000, 20000] as in rat's tasks. Human range is 20-20000 and mice 1000-70000
    # FsOut=44100 the most used (audio CD)

    ####################################################################################################################

    # Select the folder and create it if it doesn't exist
    folder = Path.home()/'Music'/'sounds_test'

    if not os.path.exists(folder):
        os.mkdir(folder)

    # 26 * 26 chars = 676 possible sounds per ild
    chars = list(string.ascii_lowercase)  # Make a list of all the lowercase letters as long as ilds

    # Create DataFrame column labels
    left_frames = [f'EL{n:01}' for n in range(n_frames)]  # Envelope Left * 10 frames
    right_frames = [f'ER{n:01}' for n in range(n_frames)]  # Envelope Right * 10 frames
    columns = (['filename', 'ILD'] +
               left_frames + right_frames +
               ['max_vol', 'fs', 'cutoff', 'amp', 'dur', 'fn', 'normalize', 'n_frames', 'sigma', 'save'])

    sound_number = 0  # Initialize counter
    data = []

    for k in range(len(ILDs_dB)):

        dB_left = dBs_left[k]
        dB_right = dBs_right[k]

        for i, j in itertools.product(chars, chars):  # Iterate through all the possible combinations of chars
            # Sound number (name) from 1 (aaa) to 9261 (uuu)
            sound_number += 1
            # name = folder + chars[k] + i + j
            name = Path(folder / (chars[k] + i + j))


            filename = chars[k] + i + j  # For the csv file

            path_wav = Path(name).with_suffix('.wav')
            print(path_wav)
            # path_mp3 = name + '.mp3'
            # path_ogg = name + '.ogg'
            # print(sound_number, name)

            SL, SR, EL, ER = do_envelope_dB_normal(noise, dB_left, dB_right, max_vol,
                                                   fs=fs, amp=amp, dur=dur, n_frames=n_frames, sigma=sigma)

            data.append([filename] + [ILDs_dB[k]] + list(EL) + list(ER) + [max_vol] + [fs] + [cutoff] + [amp] + [dur] +
                        [fn] + [normalize] + [n_frames] + [sigma] + [save])

            if save:  # Save sounds only if specified (don't wanna for simulation purposes)
                # sound = np.column_stack((filename, SL, SR))
                sound = np.column_stack((SL, SR))
                # wavio.write(path_wav, sound, fs, sampwidth=1)  # Write the array sound to a wav file
                wavio.write(path_wav.absolute().as_posix(), sound, fs, sampwidth=1)  # Write the array sound to a wav file
                # sound_wav = AudioSegment.from_wav(path_wav)  # Read the wav file to a wav sound
                # sound_wav.export(path_mp3, format='mp3')  # Export the wav sound to a mp3 file
                # sound_wav.export(path_ogg, format='ogg')  # Export the wav sound to a ogg file

            if k == 0 or k == 8:
                break  # Not to create more than 1 sound for maximum evidence

    df = pd.DataFrame(data=data, index=None, columns=columns)

    if save:
        df.to_csv(Path(folder / 'sounds_test').with_suffix('.csv'), index=False)
        # index=False to avoid writing the 'Unnamed:' column

    time_end = time.time()
    runtime = time_end - time_start
    print('\nThe script took', round(runtime, 2), 'seconds to run')

    return df
