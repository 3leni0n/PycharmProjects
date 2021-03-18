import sys
import pandas as pd
import numpy as np
import string
from my_fun.my_fun import my_select_evidence  # Atm just copy paste the function definition here

########################################################################################################################

stage = 1  # Select training stage: 0 = 'zero' = evidences[-1, 1]
#                                   1 = 'ez'   = evidences[-1:-0.75, 0.75:1]
#                                   2 = 'mid'  = evidences[-1:-0.4, 0.4:1]
#                                   3 = 'hard' = evidences[-1:0.1, 0.1:1]
#                                   4 = 'hero' = evidences[-1:1]

substage = 1

if stage == 0 or stage == 4:
    substage = 0  # Because in stages 0 and 4 there are not substages and it's index is 0
else:
    substage = substage  # Select substage [1:3]

# If stage != 0 or stage != 4, substage can't be 0, else df_sample will be empty!!!

# Select substage 1 -3. 1 easiest sounds from first character only, 2 is a transition stage with sounds
# from both characters and 3 is the hardest with sounds from the second character only
df_sample = df[(df.stage == stage) & (df.substage == substage)]
evidences_sample = df_sample.evidence.unique()  # Get evidences for a given difficulty. 'Significantly faster than numpy.unique'
thisTrialEvidence = my_select_evidence(np.random.choice([0, 1]), evidences_sample)
sample_index = df_sample.index.values
thisTrialSound = np.random.choice(sample_index)  # Randomly choose a sound from sample_index
print('Sound ', thisTrialSound, ': ', df.filename[thisTrialSound], sep='')
filename = df.filename[thisTrialSound]
keys = list(filename)  # Construct list from filename to use as dictionary keys
TTLs = [sounds_dict.get(key) for key in keys]  # Convert thisTrialSound into TTL pulses
print(TTLs)

# Legacy
########################################################################################################################

# Inclusive or exclusive sounds per difficulty? Pro of inclusive it's more progressive, cons is that by the end the prob
# of occurrence of difficult sounds is low
# Select difficulty according to stage. Basically stage and difficulty are 2 names for the same thing
if stage == 0:
    difficulty = 'zero'
    df_sample = df.loc[df['filename'].str.startswith(('a', 'u'), na=False)]  # Returns sample DataFrame
    # with sounds whose name starts with the indicated characters. na=False because there is one sound called 'nan'
    # df_sample = df[df.difficulty == 'zero']  # Same but smarter
elif stage == 1:
    difficulty = 'ez'
    if substage == 1:  # Evidence +- 0.9
        df_sample = df.loc[df['filename'].str.startswith(('b', 't'), na=False)]
    elif substage == 2:  # Evidence +- 0.8
        df_sample = df.loc[df['filename'].str.startswith(('b', 'c', 's', 't'), na=False)]
    elif substage == 3:  # Evidence +- 0.75
        df_sample = df.loc[df['filename'].str.startswith(('b', 'c', 'd', 'r', 's', 't'), na=False)]
        # df_sample = df[df.difficulty == 'ez']
elif stage == 2:
    difficulty = 'mid'
    if substage == 1:  # Evidence +- 0.6
        df_sample = df.loc[df['filename'].str.startswith(('e', 'q'), na=False)]
    elif substage == 2:  # Evidence +- 0.5
        df_sample = df.loc[df['filename'].str.startswith(('e', 'f', 'p', 'q'), na=False)]
    elif substage == 3:  # Evidence +- 0.4
        df_sample = df.loc[df['filename'].str.startswith(('e', 'f', 'g', 'o', 'p', 'q'), na=False)]
        # df_sample = df[df.difficulty == 'mid']
elif stage == 3:
    difficulty = 'hard'
    if substage == 1:  # Evidence +- 0.3
        df_sample = df.loc[df['filename'].str.startswith(('h', 'n'), na=True)]  # na=True because 'nan' belongs to hard
    elif substage == 2:  # Evidence +- 0.25
        df_sample = df.loc[df['filename'].str.startswith(('h', 'i', 'm', 'n'), na=True)]
    elif substage == 3:  # Evidence +- 0.1
        df_sample = df.loc[df['filename'].str.startswith(('h', 'i', 'j', 'l', 'm', 'n'), na=True)]
    # df_sample = df[df.difficulty == 'hard']
elif stage == 4:  # Evidence 0. As it's impossible to solve, they can't learn it, so final evidences here
    difficulty = 'hero'
    df_sample = df.loc[df['filename'].str.startswith('k', na=False)]

nTrials = 5
trialTypes = [0, 1]  # 0 (rewarded left) or 1 (rewarded right)

for trial in range(nTrials):
    thisTrialType = np.random.choice(trialTypes)  # Randomly choose trial type

    if thisTrialType == 0:  # Left trial
        print('Trial ', trial + 1, ': L', sep='')  # + 1 so there's no 'Trial 0'
        if substage == 1:
            sample_index = np.where(df['filename'].str.startswith('a', na=False))  # Return array with indexes of sounds
            # that starts with the characters specified (the first letter says the evidence)
        elif substage == 2:
            sample_index = np.where(df['filename'].str.startswith(('a', 'b'), na=False))
        elif substage == 3:
            sample_index = np.where(df['filename'].str.startswith('b', na=False))

    elif thisTrialType == 1:  # Right trial
        print('Trial ', trial + 1, ': R', sep='')
        if substage == 1:
            sample_index = np.where(df['filename'].str.startswith('r', na=False))
        elif substage == 2:
            sample_index = np.where(df['filename'].str.startswith(('r', 's'), na=False))
        elif substage == 3:
            sample_index = np.where(df['filename'].str.startswith('s', na=False))

    thisTrialSound = np.random.choice(sample_index[0])  # Randomly choose a sound from sample_index
    print('Sound ', thisTrialSound, ': ', df.filename[thisTrialSound], sep='')
    filename = df.filename[thisTrialSound]
    keys = list(filename)  # Construct list from filename to use as dictionary keys
    TTLs = [sounds_dict.get(key) for key in keys]  # Convert thisTrialSound into TTL pulses
    print(TTLs)

# df_sample = df.loc[np.where((df.difficulty == 'zero') | (df.difficulty == 'ez'))[0]]
# df_sample = df.loc[(df.difficulty == 'zero') | (df.difficulty == 'ez')]
