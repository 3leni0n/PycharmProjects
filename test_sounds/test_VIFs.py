import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load sounds
# sounds_path = '/home/alexis/PycharmProjects/create_sounds/sounds.csv'
sounds_path = '/home/alexis/PycharmProjects/create_sounds/sounds_2.csv'
sounds = pd.read_csv(sounds_path)
n_frames = 10

# Left frames
left_frames_column_names = [f'EL{n:01}' for n in range(n_frames)]
frames_left = sounds[left_frames_column_names]

# Right frames
right_frames_column_names = [f'ER{n:01}' for n in range(n_frames)]
frames_right = sounds[right_frames_column_names]

# Frames ILD (elementwise)
frames_ild = pd.DataFrame(
    sounds[right_frames_column_names].values - sounds[left_frames_column_names].values)  # Directly on the dataframe

ilds = np.sort(sounds.ILD.unique())
frames_ild = sm.add_constant(frames_ild)  # statsmodels vif function requires to add constant to design matrix first

# Test variance inflation factor (VIF)
for i in range(1, len(ilds) - 1):  # Skip extreme ILDs
    print(f'Computing VIFs of ILD = {ilds[i]}')
    ilds_index = sounds[sounds.ILD == ilds[i]].index.values  # Get indexes of sounds with a given ILD
    df_sub = frames_ild.iloc[ilds_index]  # Use those indexes to get the frames_ild with a given ILD
    # df_sub = sm.add_constant(df_sub)
    VIFs = [variance_inflation_factor(df_sub.values, i) for i in range(df_sub.shape[1])]
    VIFs = pd.Series(VIFs, index=frames_ild.columns)
    print(VIFs, '\n')
