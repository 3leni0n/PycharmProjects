"""
Notes

Bootstrapping uses brute computational force to simulate replicating an experiment, which is expensive and
time-consuming. It is an alternative to statistical inference based on parametric assumptions. Given a dataset of
size N samples, we randomly draw the same N samples with replacement (drawing without replacement will lead to the
exact same dataset from which we are drawing samples, which is pointless) to get a new, bootstrapped dataset. This
new bootstrapped dataset gives us a hint of what would look like if we would repeat the experiment. We can now
calculate statistics (mean, median, std, sem). Repeat the process many times, like 1-10k. This is bootstrapping.
Bootstrapping can be used to estimate the underlying distribution from a sample, by treating that sample like the
population from which to draw samples from. If we calculate the mean of every bootstrapped dataset and save it,
the std of that bootstrapped distribution is the same as the sem from the original dataset, and a 95% confidence
interval (CI) is just an interval that covers 95% of the bootstrapped means. If the 95% CI covers 0, we can't reject
H0 (hypothesis testing). We can apply bootstrapping to any statistc, not only the mean, without using formulas.

For calculating p-values, first calculate the mean (or any other statistic) of the dataset if the null hypothesis (H0)
was true. Then shift all the values of the dataset the H0 mean units so that the mean of the shifted data is the same as
the mean if H0 was true. Use bootstrapping to see how the mean varies under H0 varies. Make a histogram of the means and
use this distribution to get p-values.

4 steps:
1. Make a bootstrapped dataset
2. Calculate something
3. Keep track of that calculation
4 Repeat  steps 1-3 a bunch of times

Resources:
Bootstrapping Main Ideas!!! - StatQuest: https://www.youtube.com/watch?v=Xz0x-8-cgaQ
Using Bootstrapping to Calculate p-values!!! - StatQuest: https://www.youtube.com/watch?v=N4ZQQqyIf6k&t=8s
https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29
http://allendowney.blogspot.com/2011/05/there-is-only-one-test.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
http://pillowlab.princeton.edu/teaching/mathtools16/slides/lec21_Bootstrap.pdf
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

# Mel's code snippet for poster
sns.set_theme()
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')
# sns.despine()

indexes = df.reset_index().index.values

bootstrap_samples = []
time_start = time.time()

for _ in range(100):

    # 1x = 0.84s
    # 10x = 13s
    # 100x = 2 min
    # 1000x = 23 min

    # bs = np.random.choice(indexes, size=len(indexes), replace=True)
    df_bs = df.sample(frac=1, replace=True)  # Bootstrapped df

    # Get complete dataset compute every iteration, otherwise the 2nd time will be doing the half of the half!
    choices = df_bs.Choice.reset_index(drop=True)  # Indices must match for modeling
    filenames = df_bs.Filename.tolist()
    stim_strength = frames_ild.loc[
        [np.where(sounds.filename == np.array(filenames[i]))[0][0] for i in range(len(filenames))]].drop(
        columns=['filename'])
    stim_strength.reset_index(drop=True, inplace=True)  # Indices must match for modeling
    stim_strength = sm.add_constant(stim_strength)  # Add constant (bias)

    # model = sm.Logit(choices, stim_strength)  # Discrete Logit model
    model = sm.GLM(choices, stim_strength,
                   family=sm.families.Binomial())  # GLM with Binomial family and Logit link
    results = model.fit()
    params = results.params

    bootstrap_samples.append(params)


time_end = time.time()
runtime = time_end - time_start
print('The script took', round(runtime, 2), 'seconds to run')


plt.figure(constrained_layout=True)
bs_errorbar = np.std(bootstrap_samples, axis=0)[1:]



plt.figure(constrained_layout=True)
plt.plot(np.arange(1, len(params)), params.iloc[1:11], color='k', marker=None, mfc='none', mec='none', mew=0,
         ms=0)
# Without constant (bias)
plt.errorbar(np.arange(1, len(params)), params.iloc[1:11], yerr=beta_std_err.iloc[1:11], color='tab:gray',
             marker=None, fmt='none', mfc='none', mec='none', ms=0, capsize=10, alpha=0.5, label='bse')  # Without constant (bias)
plt.errorbar(np.arange(1, len(params)), params.iloc[1:11], yerr=bs_errorbar, color='tab:blue',
             marker=None, fmt='none', mfc='none', mec='none', ms=0, capsize=10, alpha=0.5, label='bs_100')  # Without constant (bias)

plt.legend(loc='best', frameon=False, fontsize='xx-small')
