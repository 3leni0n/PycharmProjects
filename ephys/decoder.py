# Neuromatch tutorial https://compneuro.neuromatch.io/tutorials/W1D5_DeepLearning/student/W1D5_Tutorial1.html


from pathlib import Path
from ephys.preprocessing import *
from ephys.analysis import *



id = '007_2024-06-23_12-46-55'
path_behavior = (Path.home() / 'Downloads' / '007_stage_training_v5_20240623-130152').with_suffix('.csv')
df_ttl, df_behavior, n_trials, df_spikes, cluster_info, x, height, labels, y, width, left, ts_edges, events_edges = \
    preprocess(id, path_behavior)


# Step 1, make multidimensional array of spiking data
# spiking data (shape: trials x binned time x neurons)
cluster = 12
df_cluster = df_spikes[df_spikes.cluster == cluster]
peri_stim_spikes = get_peri_stim_spikes(df_cluster, df_ttl, time_win=[-1, 3], scale=0)
bins, psth = compute_psth(peri_stim_spikes, time_win=[-1, 3], bin_size=0.1)












def decode_condition(X_shuffle=np.zeros((1, 1)), y=np.zeros((1, 1)), df_sess=pd.DataFrame(), \
                     number_perm=50, BALANCED=False):
    # crossvalidation x trials x time
    pred = np.empty((X_shuffle.shape[0], X_shuffle.shape[1]))
    prederr = np.empty((X_shuffle.shape[0], X_shuffle.shape[1]))

    # crossvalidate results
    # perform k-fold stratified crossvalidation
    # kf = StratifiedKFold(n_splits=5, shuffle=True) # 5-fold crossvalidation
    kf = LeaveOneOut()  # leave-one-out crossvalidation
    # split data into training and testing set
    for train_idx, test_idx in kf.split(X_shuffle, np.round(np.angle(y)).astype(str)):
        # train decoder in each time step
        for t_id, delta_t_train in enumerate(range(X_shuffle.shape[1])):
            # define train and testing set
            X_train, X_test = X_shuffle[train_idx, delta_t_train], X_shuffle[test_idx, delta_t_train]
            y_train, y_test = y[train_idx], y[test_idx]

            # train the decoder (linear algebra version)
            weights = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

            pred[test_idx, delta_t_train] = np.angle(X_test.dot(weights))
            prederr[test_idx, delta_t_train] = (circdist(np.angle(X_test.dot(weights)), \
                                                         np.angle(y_test)))

    return pred, prederr




# do this for each session separately

# z-score spiking (shape: trials x binned time x neurons)
mean_spikingpertime = np.mean(spiking, axis=0)
std_spikingpertime = np.std(spiking, axis=0)
std_spikingpertime[std_spikingpertime < 0.1] = 0.1 # choose a cut-off for where std needs to be set (in case you get 0)
X_shuffle = (spiking_prevcurr - mean_spikingpertime) / std_spikingpertime


if INTERCEPT:
    X_shuffle = np.append(np.ones((X_shuffle.shape[0], X_shuffle.shape[1], 1)), X_shuffle, axis=-1)

# call decoder
pred, prederr = decode_condition(X_shuffle = X_shuffle, y = Cue, df_sess= df)