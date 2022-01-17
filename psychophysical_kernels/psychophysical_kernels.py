"""
Notes from Genis:

The kernel stimates the weight subjects gives to each stimulus frame. It's usually computed via logistic regression
(https://en.wikipedia.org/wiki/Logistic_regression). We estimate the probability of a decision 'right' given some filters
(the betas or weights).
- p is the probability of choose right
- B0 isn't multiplied by any x and therefore is the bias. Normally is not included, but if the subject is biased, it's
best to do so. Bi are the weights of each frame, and there's one beta for each x
- x are the frames, there's one x for each B

In the wikipedia example plot, the x axis would be the stimulus strength... and the y axis would be probability of
choose right. Then we fit the logistic regression curve. When we plot a kernerl, what we're actually representing are
values of Bi. The values of beta can be computes in python with the 'logistic regression' from the 'sklearn' library
(https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- x will be a matrix with my stimulus strengths (1 row per stimulus, one column for each frame, so 1*10)
- y will be the subjects' choices

"""

import pandas as pd
import numpy as np
from parse.parse import parse
from sklearn.linear_model import LogisticRegression
import statsmodels.discrete.discrete_model as sm
from matplotlib import pyplot as plt


sounds = pd.read_csv('/create_sounds/sounds_1.csv')
sounds_left = sounds.iloc[:, 1:11]  # Index left skipping 'filename'
sounds_left.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # DataFrames needs to have BOTH the same
# row and column indices in order to perform an element-wise subtraction
sounds_right = sounds.iloc[:, 11:21]  # Index right
sounds_right.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sounds_diff = sounds_right + sounds_left  # Sum as both sides are opposite signs
sounds_diff.insert(0, column='filename', value=sounds.filename)

# df = parse('/home/alexis/pv_nmdar_eranet/experiments/2AFC/setups/910/sessions/910_stage_training_20211109-181612/910_stage_training_20211109-181612.csv')
df = pd.read_csv('/home/alexis/PycharmProjects/glue_sessions/913.csv')
df = df[df.Substage >= 5]
df = df[df.Choice.notna()]  # Drop misses
filenames = df.Filename.tolist()
stim_strength = sounds_diff.loc[[np.where(sounds.filename==np.array(filenames[i]))[0][0] for i in range(len(filenames))]].drop(columns=['filename'])
# stim_strength = sounds.loc[sounds['filename'].isin(filenames)].drop(columns=['filename'])  # Doesn't keep duplicates
choices = df.Choice.tolist()

plt.figure()

# Method 1
clf = LogisticRegression(random_state=0).fit(stim_strength, choices)
clf.get_params()
plt.plot(np.arange(len(clf.coef_[0])), clf.coef_[0], marker='o', mfc='None', label='Method 1')
# plt.title('Psychophysical kernel (Method 1)')
# plt.title('Psychophysical kernel')
# plt.xlabel('Number of frames')
# plt.ylabel('Weight')

# Method 2 - From Genis' paper analysis code (gives directly the error)
# Paper: https://www-nature-com.sire.ub.edu/articles/s41467-021-21501-z
# Code: https://bitbucket.org/delaRochaLab/flexible-categorization/src/master/functions/analysis_fc.py
logit = sm.Logit(choices, stim_strength)  # d = decisions, x = stimulus strengths
fit = logit.fit()
params = fit.params
beta_std_err = fit.bse
p_values = fit.pvalues
plt.plot(np.arange(len(params)), params, marker='o', mfc='None', label='Method 2')
plt.errorbar(np.arange(len(params)), params, yerr=beta_std_err, color='tab:blue', fmt='o', markerfacecolor='none')
# plt.title('Psychophysical kernel (Method 2)')
plt.title(f'Psychophysical kernel, animal {df.Setup.unique()[0]}, {len(df)} trials')
plt.xlabel('Number of frames')
plt.ylabel('Weight')
plt.legend()

