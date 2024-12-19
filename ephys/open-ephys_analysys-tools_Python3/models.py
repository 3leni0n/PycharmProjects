from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
import seaborn as sns

from ephys.preprocessing import *
from ephys.analysis import *

sns.set_theme()
sns.set_style('ticks')
sns.set_context('poster')


ephys_ids = ['007_2024-06-23_12-46-55',
             '007_2024-06-24_17-47-22',
             '007_2024-06-27_15-06-28']
             # '007_2024-07-09_12-10-57',
             # '007_2024-07-10_12-03-35']
behavior_ids = ['007_stage_training_v5_20240623-130152',
                '007_stage_training_v5_20240624-180217',
                '007_stage_training_v5_20240627-152129']

df = pd.DataFrame()

for i in range(len(ephys_ids)):

    id = ephys_ids[i]
    path_behavior = (Path.home() / 'Downloads' / behavior_ids[i]).with_suffix('.csv')
    df_ttl, df_behavior, n_trials, df_spikes, cluster_info = preprocess(id, path_behavior)

    # Add lick data to behavior dataframe
    """
    Qs:
    What causes the variability in the lick RT?
    What causes the variability in the lick rate (ie inter-lick-interval)?
    What makes some correct responses  have 2-3 licks and some 8-10?
    Are all response selective neurons locked to the licks?
    """
    bin_size = 0.1
    licks, n_licks, rt = get_rt(df_behavior)
    # licks = get_peri_stim_licks(df_behavior)
    bins, licks_psth = compute_psth(licks, time_win=[1, 2], bin_size=bin_size)
    df_behavior['RT'] = rt
    df_behavior['nLicks'] = n_licks
    df_behavior['LickRate'] = np.mean(licks_psth, axis=1) / bin_size

    # Add baseline FR and sync to behavior dataframe
    peri_stim_spikes = get_peri_stim_spikes(df_spikes, df_ttl, time_win=[-2, 0])
    bins, baseline_FR = compute_psth(peri_stim_spikes, time_win=[-2, 0], bin_size=bin_size)
    baseline_FR = np.mean(baseline_FR, axis=1)
    baseline_FR = baseline_FR/len(cluster_info)
    sync = get_sync(df_spikes, df_ttl, time_win=[-2, 0], bin_size=0.02, method='anal')
    df_behavior['BaselineFR'] = baseline_FR
    df_behavior['Sync'] = sync

    df = pd.concat([df, df_behavior], ignore_index=True)


# Make new column called SessionIndex
df['SessionIndex'] = df.groupby('Session').ngroup()

session_index = pd.get_dummies(df.SessionIndex, dtype='int')
n_sessions = df.Session.nunique()  # Number of sessions
df = pd.concat([df, session_index], axis=1)  # Add session index to the dataframe

# Normalize trial number and zscore baselineFR and sync (per session)
df['normTrial'] = df.groupby('Session').Trial.transform(lambda x: (x / x.max()))
# df['zSync'] = df.groupby('Session').Sync.transform(lambda x: zscore(x))
# df['zBaseFR'] = df.groupby('Session').BaselineFR.transform(lambda x: zscore(x))
df['normSync'] = df.groupby('Session').Sync.transform(lambda x: (x / x.max()))
df['normBaseFR'] = df.groupby('Session').BaselineFR.transform(lambda x: (x / x.max()))
df['normLickRate'] = df.groupby('Session').LickRate.transform(lambda x: (x / x.max()))
df['normRT'] = df.groupby('Session').RT.transform(lambda x: (x / x.max()))

########################################################################################################################

# GLMs
after_error_indexes = df[df.AfterHit == 0].index.values
after_hit_indexes = df[df.AfterHit == 1].index.values

# Accuracy (all trials)
# endog = df.Hit
endog = df.iloc[after_hit_indexes].Hit.reset_index(drop=True)
exog = pd.DataFrame({'Trial': df.normTrial, 'BaseFR': df.normBaseFR, 'Sync': df.normSync})
exog = pd.concat([exog, session_index], axis=1)
exog = exog.iloc[after_hit_indexes].reset_index(drop=True)
# exog = sm.add_constant(exog)  # Add constant (not needed if adding one intercept per session)
model = sm.GLM(endog, exog, family=sm.families.Binomial(), missing='drop')  # GLM with Binomial family
results = model.fit()
params = results.params
bse = results.bse
p_values = results.pvalues
summary = results.summary()
print(summary)
x = params.index.values[:-n_sessions]
y = params.values[:-n_sessions]
yerr = bse[:-n_sessions]
color = 'tab:green'
plt.figure(constrained_layout=True)
plt.errorbar(x, y, yerr=yerr, color=color, fmt='o')
plt.axhline(0, color='tab:gray', linestyle='--')
plt.title(f'Accuracy ({n_sessions} sessions, {len(endog)} trials)')
# plt.xlabel('Coefficients')
plt.ylabel('Weight')
# plt.legend(frameon=False)
sns.despine()

# Plot intercepts
plt.errorbar(np.repeat(len(params[:-n_sessions]), n_sessions), params.values[-n_sessions:], yerr=bse[-n_sessions:],
             color=color, fmt='o')
xticks = plt.xticks()[0] + [len(params[:-n_sessions])]
xticklabels = [label.get_text() for label in plt.xticks()[1]] + ['Cons']
plt.xticks(xticks, xticklabels)


n_shuffles = 10000  # Number of shuffles
shuffled_params = []

for _ in range(n_shuffles):
    # Shuffle the dependent variable
    shuffled_endog = endog.sample(frac=1, random_state=None).reset_index(drop=True)

    # Fit the GLM model with the shuffled data
    model = sm.GLM(shuffled_endog, exog, family=sm.families.Binomial(), missing='drop')
    results = model.fit()

    # Store the coefficients (excluding session intercepts if present)
    shuffled_params.append(results.params.values[:-n_sessions])

# Convert to a NumPy array for easier manipulation
shuffled_params = np.array(shuffled_params)

lower_bound = np.percentile(shuffled_params, 2.5, axis=0)
upper_bound = np.percentile(shuffled_params, 97.5, axis=0)
plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.25, edgecolor='none', label='Shuffle Confidence Band')







factorial(n_sessions)
from itertools import permutations

df_copy = df.copy()
# Find the number of trials per session and crop it to the minimum
n_trials = df_copy.groupby('Session').Trial.count()
min_trials = n_trials.min()
# Drop trials above the minimum for each session
df_copy = df_copy.groupby('Session').apply(lambda x: x[x.Trial < min_trials]).reset_index(drop=True)

df_test


# Make a toy data df with 3 sessions and 3 trials per session
toy_data = pd.DataFrame({
    "Session": [0] * 3 + [1] * 3 + [3] * 3,
    "Trial": [0, 1, 2] * 3,  # Repeating trial numbers across sessions
    "x": np.random.rand(9),  # Random data for variable x
    "y": np.random.rand(9)   # Random data for variable y
})

df_test = pd.DataFrame()
df_temp = toy_data.copy()
df_temp.y = np.roll(df_temp.y, 3)
df_test = pd.concat([df_test, df_temp])






















# Misses
endog = df.Miss
exog = pd.DataFrame({'Trial': df.normTrial, 'BaseFR': df.normBaseFR, 'Sync': df.normSync})
exog = pd.concat([exog, session_index], axis=1)
# exog = sm.add_constant(exog)  # Add constant (not needed if adding one intercept per session)
model = sm.GLM(endog, exog, family=sm.families.Binomial(), missing='drop')  # GLM with Binomial family
results = model.fit()
params = results.params
bse = results.bse
p_values = results.pvalues
summary = results.summary()
print(summary)
x = params.index.values[:-n_sessions]
y = params.values[:-n_sessions]
yerr = bse[:-n_sessions]
color = 'k'
plt.figure(constrained_layout=True)
plt.errorbar(x, y, yerr=yerr, color=color, fmt='o')
plt.axhline(0, color='tab:gray', linestyle='--')
plt.title(f'Misses ({n_sessions} sessions, {len(endog)} trials)')
# plt.xlabel('Coefficients')
plt.ylabel('Weight')
# plt.legend(frameon=False)
sns.despine()

# Plot intercepts
plt.errorbar(np.repeat(len(params[:-n_sessions]), n_sessions), params.values[-n_sessions:], yerr=bse[-n_sessions:],
             color=color, fmt='o')
xticks = plt.xticks()[0] + [len(params[:-n_sessions])]
xticklabels = [label.get_text() for label in plt.xticks()[1]] + ['Cons']
plt.xticks(xticks, xticklabels)


n_shuffles = 10000  # Number of shuffles
shuffled_params = []

for _ in range(n_shuffles):
    # Shuffle the dependent variable
    shuffled_endog = endog.sample(frac=1, random_state=None).reset_index(drop=True)

    # Fit the GLM model with the shuffled data
    model = sm.GLM(shuffled_endog, exog, family=sm.families.Binomial(), missing='drop')
    results = model.fit()

    # Store the coefficients (excluding session intercepts if present)
    shuffled_params.append(results.params.values[:-n_sessions])

# Convert to a NumPy array for easier manipulation
shuffled_params = np.array(shuffled_params)

lower_bound = np.percentile(shuffled_params, 2.5, axis=0)
upper_bound = np.percentile(shuffled_params, 97.5, axis=0)
plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.25, edgecolor='none', label='Shuffle Confidence Band')



# Rep. bias
endog = df.iloc[after_hit_indexes].RepChoice.reset_index(drop=True)
exog = pd.DataFrame({'Trial': df.normTrial, 'PrevOut': df.AfterHit, 'RepTrial': df.RepTrial, 'BaseFR': df.normBaseFR,
                     'Sync': df.Sync})
exog = pd.concat([exog, session_index], axis=1)
exog = exog.iloc[after_hit_indexes].reset_index(drop=True)
# exog = sm.add_constant(exog)  # Add constant (not needed if adding one intercept per session)
model = sm.GLM(endog, exog, family=sm.families.Binomial(), missing='drop')  # GLM with Binomial family
results = model.fit()
params = results.params
bse = results.bse
p_values = results.pvalues
summary = results.summary()
print(summary)
print(summary)
x = params.index.values[:-n_sessions]
y = params.values[:-n_sessions]
yerr = bse[:-n_sessions]
color = 'tab:green'
plt.figure(constrained_layout=True)
plt.errorbar(x, y, yerr=yerr, color=color, fmt='o')
plt.axhline(0, color='tab:gray', linestyle='--')
plt.title(f'Repeating bias ({n_sessions} sessions, {len(endog)} trials)')
# plt.xlabel('Coefficients')
plt.ylabel('Weight')
# plt.legend(frameon=False)
sns.despine()

# Plot intercepts
plt.errorbar(np.repeat(len(params[:-n_sessions]), n_sessions), params.values[-n_sessions:], yerr=bse[-n_sessions:],
             color=color, fmt='o')
xticks = plt.xticks()[0] + [len(params[:-n_sessions])]
xticklabels = [label.get_text() for label in plt.xticks()[1]] + ['Cons']
plt.xticks(xticks, xticklabels)
plt.xticks(rotation=45)


n_shuffles = 10000  # Number of shuffles
shuffled_params = []

for _ in range(n_shuffles):
    # Shuffle the dependent variable
    shuffled_endog = endog.sample(frac=1, random_state=None).reset_index(drop=True)

    # Fit the GLM model with the shuffled data
    model = sm.GLM(shuffled_endog, exog, family=sm.families.Binomial(), missing='drop')
    results = model.fit()

    # Store the coefficients (excluding session intercepts if present)
    shuffled_params.append(results.params.values[:-n_sessions])

# Convert to a NumPy array for easier manipulation
shuffled_params = np.array(shuffled_params)

lower_bound = np.percentile(shuffled_params, 2.5, axis=0)
upper_bound = np.percentile(shuffled_params, 97.5, axis=0)
plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.3, edgecolor='none', label='Shuffle Confidence Band')



# Lick rate
endog = df.normLickRate
# endog = df.iloc[after_hit_indexes].normLickRate.reset_index(drop=True)
exog = pd.DataFrame({'Trial': df.normTrial, 'Hit': df.Hit, 'BaseFR': df.normBaseFR, 'Sync': df.normSync})
exog = pd.concat([exog, session_index], axis=1)
# exog = exog.iloc[after_hit_indexes].reset_index(drop=True)
# exog = sm.add_constant(exog)  # Add constant (not needed if adding one intercept per session)
model = sm.GLM(endog, exog, family=sm.families.Binomial(), missing='drop')  # GLM with Binomial family
results = model.fit()
params = results.params
bse = results.bse
p_values = results.pvalues
summary = results.summary()
print(summary)
x = params.index.values[:-n_sessions]
y = params.values[:-n_sessions]
yerr = bse[:-n_sessions]
color = 'k'
plt.figure(constrained_layout=True)
plt.errorbar(x, y, yerr=yerr, color=color, fmt='o')
plt.axhline(0, color='tab:gray', linestyle='--')
plt.title(f'Lick Rate ({n_sessions} sessions, {len(endog)} trials)')
# plt.xlabel('Coefficients')
plt.ylabel('Weight')
# plt.legend(frameon=False)
sns.despine()

# Plot intercepts
plt.errorbar(np.repeat(len(params[:-n_sessions]), n_sessions), params.values[-n_sessions:], yerr=bse[-n_sessions:],
             color=color, fmt='o')
xticks = plt.xticks()[0] + [len(params[:-n_sessions])]
xticklabels = [label.get_text() for label in plt.xticks()[1]] + ['Cons']
plt.xticks(xticks, xticklabels)


n_shuffles = 10000  # Number of shuffles
shuffled_params = []

for _ in range(n_shuffles):
    # Shuffle the dependent variable
    shuffled_endog = endog.sample(frac=1, random_state=None).reset_index(drop=True)

    # Fit the GLM model with the shuffled data
    model = sm.GLM(shuffled_endog, exog, family=sm.families.Binomial(), missing='drop')
    results = model.fit()

    # Store the coefficients (excluding session intercepts if present)
    shuffled_params.append(results.params.values[:-n_sessions])

# Convert to a NumPy array for easier manipulation
shuffled_params = np.array(shuffled_params)

lower_bound = np.percentile(shuffled_params, 2.5, axis=0)
upper_bound = np.percentile(shuffled_params, 97.5, axis=0)
plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.25, edgecolor='none', label='Shuffle Confidence Band')



def plot_sync_hist(df, hue='Hit'):
    """
    Plot histogram of sync split by condition
    :param hue: 'Miss', 'RepChoice', 'Hit'
    :return:
    """

    if hue == 'Miss':
        palette = ['tab:gray', 'k']
        labels = ['Miss', 'Resp.']
    elif hue == 'RepChoice':
        palette = ['tab:purple', 'tab:brown']
        labels = ['Rep.', 'Alt.']
    elif hue == 'Hit':
        palette = ['tab:red', 'tab:green']
        labels = ['Hit', 'Error']

    # Make histogram of sync split by miss/response
    plt.figure(constrained_layout=True)
    sns.histplot(data=df, x='Sync', hue=hue, multiple='layer', kde=True, stat='density', bins='auto', common_norm=False,
                 palette=palette)
    plt.title('Sync distribution')
    sns.despine()
    plt.legend(labels=labels, frameon=False, loc='upper right')