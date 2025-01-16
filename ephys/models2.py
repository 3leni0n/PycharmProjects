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
             # '007_2024-06-22_10-48-57']

behavior_ids = ['007_stage_training_v5_20240623-130152',
                '007_stage_training_v5_20240624-180217',
                '007_stage_training_v5_20240627-152129']
                # '007_stage_training_v5_20240622-110354']


PARAMS = pd.DataFrame()
n_trials = 0

for i in range(len(ephys_ids)):

    df = pd.DataFrame()

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

    # zscore regressors (per session)
    df['zTrial'] = zscore(df.Trial)
    df['zSync'] = zscore(df.Sync)
    df['zBaseFR'] = zscore(df.BaselineFR)
    df['zLickRate'] = zscore(df.LickRate)
    df['zRT'] = zscore(df.RT)

    n_trials += len(df)


    # GLM
    endog = df.Hit
    exog = pd.DataFrame({'Trial': df.zTrial, 'BaseFR': df.zBaseFR, 'Sync': df.zSync})
    exog = sm.add_constant(exog)  # Add constant (not needed if adding one intercept per session)
    model = sm.GLM(endog, exog, family=sm.families.Binomial(), missing='drop')  # GLM with Binomial family
    results = model.fit()
    params = results.params

    PARAMS = pd.concat([PARAMS, params], axis=1)



