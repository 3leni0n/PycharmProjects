from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore
# SlidingEstimator = diagonal (same time decoding)
# GeneralizingEstimator = also off-diagonal (different time decoding)
import time
from ephys.analysis import *

# mne allows for parallelize operations. The con with sklearn is that you need to loop over time points


bins, all_psth = get_all_psth(cluster_info, df_spikes, df_ttl, n_trials, time_win=[-1, 3], bin_size=0.1)


start = time.time()
clf = LogisticRegression()
pipe = make_pipeline(StandardScaler(), clf)  # First we scale and then we classify
# mne_estimator = SlidingEstimator(pipe, n_jobs=1, scoring='accuracy', verbose=False)  # Diagonal
mne_estimator = GeneralizingEstimator(pipe, n_jobs=1, verbose=False)  # Full
# score, permutation_scores, pvalue = permutation_test_score(mne_estimator, all_psth, df_behavior.Side, cv=5, n_permutations=100, n_jobs=1)
score = cross_val_multiscore(mne_estimator, all_psth, df_behavior.Side, cv=5, n_jobs=-1)
end = time.time()
print(f"Elapsed time: {end - start:.2f} s")
