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
behavior_ids = ['007_stage_training_v5_20240623-130152.csv',
                '007_stage_training_v5_20240624-180217.csv',
                '007_stage_training_v5_20240627-152129.csv']

df = pd.DataFrame()

for i in range(len(ephys_ids)):

    id = ephys_ids[i]
    path_behavior = Path.home() / 'Downloads' / behavior_ids[i]
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
    baseline_FR = zscore(baseline_FR)
    sync = get_sync(df_spikes, df_ttl, time_win=[-2, 0], bin_size=0.02, method='anal')
    df_behavior['BaselineFR'] = baseline_FR
    df_behavior['Sync'] = sync

    df = pd.concat([df, df_behavior], ignore_index=True)

session_index = pd.get_dummies(df.Session, dtype='int')
n_sessions = df.Session.nunique()  # Number of sessions
df = pd.concat([df, session_index], axis=1)  # Add session index to the dataframe

# Normalize trial number and zscore baselineFR and sync (per session)
df['normTrial'] = df.groupby('Session').Trial.transform(lambda x: (x / x.max()))
# df['zSync'] = df.groupby('Session').Sync.transform(lambda x: zscore(x))
# df['zBaseFR'] = df.groupby('Session').BaselineFR.transform(lambda x: zscore(x))
df['zSync'] = df.groupby('Session').Sync.transform(lambda x: (x / x.max()))
df['zBaseFR'] = df.groupby('Session').BaselineFR.transform(lambda x: (x / x.max()))

########################################################################################################################

# GLMs

# Accuracy (all trials)
endog = df.Hit
exog = pd.DataFrame({'normTrial': df.Trial, 'zBaseFR': df.BaselineFR, 'zSync': df.Sync})
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


# Misses
endog = df.Miss
exog = pd.DataFrame({'normTrial': df.Trial, 'zBaseFR': df.BaselineFR, 'zSync': df.Sync})
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
endog = df.RepChoice
exog = pd.DataFrame({'normTrial': df.Trial, 'PrevOut': df.AfterHit, 'RepTrial': df.RepTrial, 'zBaseFR': df.BaselineFR,
                     'zSync': df.Sync})
exog = pd.concat([exog, session_index], axis=1)
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
color = 'tab:brown'
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
