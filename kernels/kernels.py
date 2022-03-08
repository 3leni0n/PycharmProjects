"""
Notes from Genis:

The kernel stimates the weight subjects gives to each stimulus frame. It's usually computed via logistic regression
(https://en.wikipedia.org/wiki/Logistic_regression). We estimate the probability of a decision 'right' given some filters
(the betas or weights).
- p is the probability of choose right
- B0 isn't multiplied by any x and therefore is the bias. Normally is not included, but if the subject is biased, it's
best to do so. Bi are the weights of each frame, and there's one beta for each x
- x are the frames, there's one x for each B

In the wikipedia example plot, the x-axis would be the stimulus strength and the y-axis would be probability of
choose right. Then we fit the logistic regression curve. When we plot a kernel, what we're actually representing are
values of Bi. The values of beta can be computed in python with the 'logistic regression' from the 'sklearn' library
(https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- x will be a matrix with my stimulus strengths (1 row per stimulus, one column for each frame, so 1*10)
- y will be the subjects' choices
"""


import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from matplotlib import pyplot as plt


def kernel(library='sm'):

    folder = '/home/alexis/Documentos/kernels/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    os.chdir(folder)

    # Load sounds
    # sounds_path = '/home/alexis/PycharmProjects/create_sounds/sounds.csv'
    sounds_path = '/home/alexis/PycharmProjects/create_sounds/sounds_2.csv'
    sounds = pd.read_csv(sounds_path)
    n_frames = 10

    # Left frames
    left_frames_column_names = [f'EL{n:01}' for n in range(n_frames)]
    frames_left = sounds[left_frames_column_names]
    frames_left_array = np.array(np.array(frames_left))

    # Right frames
    right_frames_column_names = [f'ER{n:01}' for n in range(n_frames)]
    frames_right = sounds[right_frames_column_names]
    frames_right_array = np.array(frames_right)

    # Frames ILD (elementwise)
    frames_ild = pd.DataFrame(sounds[right_frames_column_names].values - sounds[left_frames_column_names].values)  # Directly on the dataframe
    frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert filenames in first column
    frames_ild_array = frames_right_array - frames_left_array  # Get the column wise difference between right and left frames

    # Frames mean (elementwise)
    sounds_concat = pd.concat((pd.DataFrame(frames_left.values), pd.DataFrame(frames_right.values)))  # DataFrame concatenating left and right frames
    sounds_concat_indices = sounds_concat.groupby(sounds_concat.index)
    frames_mean = sounds_concat_indices.mean()
    frames_mean.insert(0, column='filename', value=sounds.filename)  # Insert filenames in first column
    frames_mean_array = np.mean([frames_left_array, frames_right_array], axis=0)  # Get the column wise mean between right and left frames

    # Behavioral data
    subject_path = '/home/alexis/PycharmProjects/glue_sessions/2AFC_2/335.csv'
    df = pd.read_csv(subject_path)
    df = df[df.Choice.notna()]  # Drop misses (nan in choices), otherwise the code crashes
    ilds = np.sort(df.ILD.unique())
    # target_ilds = [-2, 0, 2]
    # df = df[df.ILD.isin(target_ilds)]  # Select only trials with the desired ILDs
    filenames = df.Filename.tolist()
    stim_strength = frames_ild.loc[[np.where(sounds.filename == np.array(filenames[i]))[0][0] for i in range(len(filenames))]].drop(columns=['filename'])
    # stim_strength = sounds.loc[sounds['filename'].isin(filenames)].drop(columns=['filename'])  # Doesn't keep duplicates
    stim_strength.reset_index(drop=True, inplace=True)  # Indices must match for modeling
    choices = df.Choice.reset_index(drop=True)  # Indices must match for modeling


    # Plots
    plt.figure()

    if library=='sklearn':  # Scikit-learn library
        clf = LogisticRegression(random_state=0).fit(stim_strength, choices)
        clf.get_params()
        plt.plot(np.arange(len(clf.coef_[0])), clf.coef_[0], marker='o', mfc='None', label='Method 1')
        # plt.title('Psychophysical kernel')
        # plt.title('Psychophysical kernel')
        # plt.xlabel('Number of frames')
        # plt.ylabel('Weight')

    elif library=='sm':  # Statsmodels library
        # From Genis' paper analysis code (gives directly the error)
        # Paper: https://www-nature-com.sire.ub.edu/articles/s41467-021-21501-z
        # Code: https://bitbucket.org/delaRochaLab/flexible-categorization/src/master/functions/analysis_fc.py
        # GLM with Binomial family and Logit link = discrete Logit model
        stim_strength = sm.add_constant(stim_strength)  # Add constant (bias)
        # model = sm.Logit(choices, stim_strength)  # Discrete Logit model
        model = sm.GLM(choices, stim_strength, family=sm.families.Binomial())  # GLM with Binomial family and Logit link
        results = model.fit()
        params = results.params
        beta_std_err = results.bse
        p_values = results.pvalues
        summary = results.summary()
        print(summary)
        # plt.plot(np.arange(len(params)), params, marker='o', mfc='None', label='')  # Without constant
        # plt.errorbar(np.arange(len(params)), params, yerr=beta_std_err, color='tab:blue', fmt='o',
        #              markerfacecolor='none')  # Without constant
        plt.plot(np.arange(len(params)-1), params.iloc[1:11], color='k', marker='o', mfc='None', label='')  # With constant
        plt.errorbar(np.arange(len(params)-1), params.iloc[1:11], yerr=beta_std_err.iloc[1:11], color='k', fmt='o',
                     markerfacecolor='none')  # With constant
        plt.axhline(0, color='tab:gray', ls='--')
        plt.title(f'Psychophysical kernel, animal {df.Setup.unique()[0]}, {len(df)} trials, ILDs: {target_ilds}')
        plt.xlabel('Number of frames')
        plt.ylabel('Weight')
        # plt.legend()
        yticks = plt.gca().get_yticks()  # Get current axis yticks

        for i in range(n_frames):
            if p_values[0+i] <= 0.05:
                text = '*'
                plt.annotate(text, xy=(i, yticks[1]), xytext=(i, yticks[1]), color='k', va='center', ha='center',
                             fontsize='medium')
            else:
                text = 'ns'
            # plt.annotate(str(round(p_values[0+i], 2)),
            #              xy=(i, yticks[1]), xytext=(i, yticks[1]), color='k',
            #              va='top', ha='center', fontsize='medium')
            # plt.annotate(text, xy=(i, yticks[1]), xytext=(i, yticks[1]), color='k', va='center', ha='center', fontsize='medium')

        plt.savefig(folder + str(df.Setup.unique()[0]) + '_PK.png')



########################################################################################################################

# Legacy
stim_strength2 = stim_strength.iloc[:, 5:6]
stim_strength2 = sm.add_constant(stim_strength2)  # Add constant (bias)
model2 = sm.GLM(choices, stim_strength2, family=sm.families.Binomial())  # Binomial as there's 2 choices. No lapses because is between 0 and 1
results2 = model2.fit()
print(results2.summary())


first_frames = []
last_frames = []

for i in range(len(sounds)):
    first_frames.append(frames_ild.iloc[i, 1:6].mean())
    last_frames.append(frames_ild.iloc[i, 6:11].mean())
