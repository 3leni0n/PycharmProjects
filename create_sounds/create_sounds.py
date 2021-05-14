# Import libraries
import numpy as np
import sys  # Module that provides access to interpreter variables and functions
# sys.path.insert(1, '/home/alexis/Bitbucket/PycharmProjects/toolsR')  # Tell python to look this directory besides the cd
from toolsR import UtilsR
import os
import itertools
import wavio
from pydub import AudioSegment
import pandas as pd
from matplotlib import pyplot as plt
import string

########################################################################################################################

# Generate white noise
#whiteNoise = UtilsR.whiteNoiseGen(1.0, 2000, 20000, 1, FsOut=44100, Fn=10000, randgen=None)
whiteNoise = UtilsR.whiteNoiseGen(1.0, 2000, 20000, 1, FsOut=44100, Fn=10000, randgen=None)

# band_fs=[2000, 20000] as in rat's tasks. Human range is 20-20000 and mice 1000-70000
# FsOut=44100 the most used (audio CD)

########################################################################################################################

# Generate evidences and coherences (aim for 9 data points psychometric curves)
evidences = np.array([-1, -0.9, -0.8, -0.75, -0.6, -0.5, -0.4, -0.3, -0.25, -0.1,
                      0, 0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1])
# Evidences spaced 0.1 because len(np.arange(-1, 1, 0.1)) < len(list(string.ascii_lowercase)). In words, they can be
# expressed only with letters (20 characters), while an spacing of 0.05 would require 40 characters and use numbers too
# Evidences finished in .05 for the final task and psychometric curves
coherences = (evidences + 1) / 2  # From 0 (left) to 1 (right) so 0 net evidence returns 0.5. Input argument needed to
# calculate beta distribution in envelope function

########################################################################################################################

# Select the folder and create it if it doesn't exists
folder = '/home/alexis/Música/sounds/'

if not os.path.exists(folder):
    os.mkdir(folder)

# 21 * 21 chars = 441 possible sounds per evidence
# chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u']

# More elegant:
chars = list(string.ascii_lowercase[:len(evidences)])  # Make a list of all the lowercase letters as long as evidences

# Create DataFrame column labels
columns = ['EL0', 'EL1', 'EL2', 'EL3', 'EL4', 'EL5', 'EL6', 'EL7', 'EL8', 'EL9',  # Envelope Left * 10 frames
           'ER0', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6', 'ER7', 'ER8', 'ER9']  # Envelope Right * 10 frames

df = pd.DataFrame(data=None, index=None, columns=columns)  # Create empty data frame with column labels

sound_number = 0  # Initialize counter

for k in range(len(chars)):
    for i, j in itertools.product(chars, chars):  # Iterate through all the possible combinations of chars
        # Sound number (name) from 1 (aaa) to 9261 (uuu)
        sound_number += 1
        name = folder + chars[k] + i + j

        filename = chars[k] + i + j  # for the CSV file

        path_wav = name + '.wav'
        #path_mp3 = name + '.mp3'
        #path_ogg = name + '.ogg'
        print(sound_number, name)

        SL, SR, EL, ER = UtilsR.envelope(coherences[k], whiteNoise, dur=1, nframes=10, samplingR=44100, variance=0.015,
                                         randomized=False, paired=False, LAmp=1.0, RAmp=1.0, oldbug=False, randgen=None)

        ELER = np.concatenate((EL, ER))  # Concatenate EL and ER
        df2 = pd.DataFrame([ELER], index=[filename], columns=columns)  # Fill data frame
        df = df.append(df2)  # Append last row of data to existing data frame

        sound = np.column_stack((SL, SR))
        # Write the array sound to a wav file
        wavio.write(path_wav, sound, 44100, sampwidth=1)
        # Read the wav file to a wav sound
        sound_wav = AudioSegment.from_wav(path_wav)
        # Export the wav sound to a mp3 file
        #sound_wav.export(path_mp3, format='mp3')
        #sound_wav.export(path_ogg, format='ogg')

df.index.name = 'filename'  # Change index name from 'Unnamed: 0' to 'filename'
df.to_csv('sounds.csv')  # Save df as csv file
