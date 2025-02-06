# Neuromatch tutorial https://compneuro.neuromatch.io/tutorials/W1D5_DeepLearning/student/W1D5_Tutorial1.html


from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ephys.preprocessing import *
from ephys.analysis import *



id = '007_2024-06-23_12-46-55'
path_behavior = (Path.home() / 'Downloads' / '007_stage_training_v5_20240623-130152').with_suffix('.csv')
df_ttl, df_behavior, n_trials, df_spikes, cluster_info, x, height, labels, y, width, left, ts_edges, events_edges = \
    preprocess(id, path_behavior)




def get_all_psth(group, df_spikes, cluster_info, time_win=[-1, 3], bin_size=0.1):
    """
    Create 3-dimensional array (trial x bin x cluster) with all PSTHs for all clusters
    :param df_spikes: dataframe with spike times
    :param cluster_info: dataframe with cluster information
    :param time_win: time window around event
    :param bin_size: bin size
    :return: bins, all_psth
    """

    if group == 'all':
        cond = (cluster_info.group == 'good') | (cluster_info.group == 'mua')
    elif group == 'good':
        cond = cluster_info.group == 'good'
    elif group == 'mua':
        cond = cluster_info.group == 'mua'

    n_bins = int((time_win[1] - time_win[0]) / bin_size) + 1
    # n_clusters = len(cluster_info[cluster_info.group == 'good'])
    n_clusters = len(cluster_info[cond])


    all_psth = np.zeros((n_trials, n_bins, n_clusters))  # Initialize array to store all PSTHs

    for i, cluster in enumerate(cluster_info[cluster_info.group == 'good'].cluster_id):
        print(f'{i}: cluster {cluster}')
        df_cluster = df_spikes[df_spikes.cluster == cluster]
        peri_stim_spikes = get_peri_stim_spikes(df_cluster, df_ttl, time_win=[-1, 3], scale=0)
        bins, psth = compute_psth(peri_stim_spikes, time_win=[-1, 3], bin_size=0.1)
        all_psth[:, :, i] = psth

    return bins, all_psth
























def decode_condition(X=np.zeros((1, 1)), y=np.zeros((1, 1))):
    # crossvalidation x trials x time
    pred = np.empty((X.shape[0], X.shape[1]))
    prederr = np.empty((X.shape[0], X.shape[1]))

    # crossvalidate results
    # perform k-fold stratified crossvalidation
    # kf = StratifiedKFold(n_splits=5, shuffle=True) # 5-fold crossvalidation
    kf = KFold()  # leave-one-out crossvalidation
    # split data into training and testing set
    for train_idx, test_idx in kf.split(X):
        # train decoder in each time step
        for t_id, delta_t_train in enumerate(range(X.shape[1])):
            # define train and testing set
            X_train, X_test = X[train_idx, delta_t_train], X[test_idx, delta_t_train]
            y_train, y_test = y[train_idx], y[test_idx]

            # train the decoder (linear algebra version)
            clf = LogisticRegression()
            clf.fit(X_train, y_train)

            # Evaluate
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {acc:.2f}")

            pred[test_idx, delta_t_train] = y_pred
            prederr[test_idx, delta_t_train] = y_pred - y_test
    return pred, prederr


pred, prederr = decode_condition(X=all_psth, y=df_behavior.Side)

plt.plot(-np.mean(abs(prederr), axis=0)+1)



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