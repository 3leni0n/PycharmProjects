import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import sem
from my_fun.my_fun import timer
from ephys.preprocessing import *
from ephys.analysis import *
import seaborn as sns
import time
import pickle

sns.set_theme()
sns.set_style('ticks')
sns.set_context('talk')

# Neuromatch tutorial: https://compneuro.neuromatch.io/tutorials/W1D5_DeepLearning/student/W1D5_Tutorial1.html

########################################################################################################################

subject = '007'
folder_parent = Path.home() / 'data' / subject

ephys_ids = [
    '007_2024-06-22_10-48-57',
    '007_2024-06-23_12-46-55',
    '007_2024-06-24_17-47-22',
    '007_2024-06-25_15-54-23',
    '007_2024-06-26_15-19-30',
    '007_2024-06-27_15-06-28',
    '007_2024-06-28_14-18-51',
    '007_2024-06-29_16-24-52',
    # '007_2024-07-06_11-16-25',  # Bad session
    # '007_2024-07-07_13-20-29',  # Bad session
    '007_2024-07-08_12-20-29',
    '007_2024-07-09_12-10-57',
    '007_2024-07-10_12-03-35',
    '007_2024-07-11_12-39-21',
    '007_2024-07-12_13-29-26'
]

# Compute spike counts for all neurons of a session (bins, all_psth). If file doesn't exist, create it
for i in range(len(ephys_ids)):

    print(f'Processing session {i + 1}/{len(ephys_ids)}: {ephys_ids[i]}...')

    # Create child folder within parent folder for each ephys_id with its name if it doesn't exist
    folder_child = folder_parent / ephys_ids[i]
    folder_child.mkdir(parents=True, exist_ok=True)
    os.chdir(folder_child)

    # Get spike counts
    if any(f.endswith('.npy') for f in os.listdir(folder_child)):
        print('Files exist in folder. Skipping')
        continue
    else:
        print('Files do not exist in folder. Proceeding')
        df_ttl, df_behavior, n_trials, df_spikes, cluster_info, x, height, labels, y, width, left, ts_edges, events_edges = \
            preprocess(ephys_ids[i])
        bins, all_psth = get_all_psth(cluster_info, df_spikes, df_ttl, n_trials, time_win=[-1, 3], bin_size=0.1)
    # Save bins and psth
    np.save(folder_child / 'bins.npy', bins)
    np.save(folder_child / 'all_psth.npy', all_psth)


# Parse behavior of a session. If file doesn't exist, create it
for i in range(len(ephys_ids)):

    path_behavior = get_behavior_id(ephys_ids[i])
    filename = path_behavior.name
    print(f'Parsing behavioral session {i + 1}/{len(ephys_ids)}: {filename[:-4]}...')

    # Create child folder within parent folder for each ephys_id with its name if it doesn't exist
    folder_child = folder_parent / ephys_ids[i]
    folder_child.mkdir(parents=True, exist_ok=True)
    os.chdir(folder_child)

    # Execute only if folder is empty
    if any(f.endswith('.csv') for f in os.listdir(folder_child)):
        print('File exist in folder. Skipping')
        continue
    else:
        print('File does not exist in folder. Proceeding')
        df_behavior = parse_v2(path_behavior)
        df_behavior.to_csv(folder_child / filename, index=False)
        # behavior.append(df_behavior)


behavior = []
for i in range(len(ephys_ids)):
    path_behavior = get_behavior_id(ephys_ids[i])
    filename = path_behavior.name
    print(i, filename)
    folder_child = folder_parent / ephys_ids[i]
    df = pd.read_csv(folder_child / filename)
    behavior.append(df)


def find_disengaged(df_behavior, threshold=0.5, min_trial=200, win_len=20, plot=False):
    """
    Find the first trial where the animal disengages from the task based on side accuracy.
    :param df_behavior: DataFrame with behavioral data
    :param threshold: threshold accuracy to consider the animal disengaged
    :param min_trial: minimum trial to start looking for disengagement
    :param win_len: window length to compute rolling average
    :return: first_trial (int)
    """

    x_total, y_total, x_0, y_0, x_1, y_1 = get_roll_avg(df_behavior, kind='side')

    # Convert indices to lists to ensure compatibility
    x_0, x_1, x_total = list(x_0), list(x_1), list(x_total)

    # Adjust minimum trial to account for the running window
    min_valid_trial = min_trial + win_len

    # Filter trials starting from min_valid_trial
    filtered_x_total = [(x, y) for x, y in zip(x_total, y_total) if x >= min_valid_trial]
    filtered_x_0 = [(x, y) for x, y in zip(x_0, y_0) if x >= min_valid_trial]
    filtered_x_1 = [(x, y) for x, y in zip(x_1, y_1) if x >= min_valid_trial]

    # Find first trial where y_total reaches threshold
    idx_total = next(((x, y) for x, y in filtered_x_total if y <= threshold), None)

    # Find first trial where y_0 or y_1 reaches threshold, mapped back to absolute trials
    idx_0 = next(((x_total[x_total.index(x)], y) for x, y in filtered_x_0 if y <= threshold), None) if filtered_x_0 else None
    idx_1 = next(((x_total[x_total.index(x)], y) for x, y in filtered_x_1 if y <= threshold), None) if filtered_x_1 else None

    # Get the earliest occurrence and corresponding y-value
    disengaged_trial, disengaged_y = min(filter(None, [idx_total, idx_0, idx_1]), default=(None, None))

    print(f'Disengagement happened in trial {disengaged_trial}')

    # Side accuracy plot
    if plot:
        plt.figure(constrained_layout=True)
        plt.plot(x_total, y_total, color='k')
        plt.plot(x_0, y_0, color='tab:blue')
        plt.plot(x_1, y_1, color='tab:orange')
        plt.axhline(0.25, color='tab:gray', ls=':')
        plt.axhline(0.5, color='tab:gray', ls='--')
        plt.axhline(0.75, color='tab:gray', ls=':')

        # Plot the red dot at the correct x (absolute trial number) and y=0.5
        if disengaged_trial is not None:
            plt.plot(disengaged_trial, threshold, 'ro')

        plt.xlabel('Trial')
        plt.ylabel('Side acc.')
        plt.title(df_behavior.Session.unique()[0])
        sns.despine()

    return disengaged_trial


# for i in range(len(behavior)):
#     disengagement = find_disengaged(behavior[i], plot=False)  # Find trial when disengagement happens
#     engaged = (behavior[i]['Trial'] <= disengagement).astype(int)  # Compare trials to disengagement (no need for i)
#     behavior[i]['Engaged'] = engaged

# # Get behavioral events
# stim_dur = df_behavior.StimDur.unique()[0]
# delay = df_behavior.Delay.unique()[0]
# go_cue = stim_dur + delay

########################################################################################################################
# SINGLE SESSION DECODERS
########################################################################################################################


@timer
def within_decoder(X=np.zeros((1, 1, 1)), y=np.zeros((1, 1)), n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary stimulus condition from neural data using K-fold
    cross-validation across trials and across time bins.
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary stimulus condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape
    pred = np.empty((n_trials, n_bins))
    pred_err = np.empty((n_trials, n_bins))
    acc = np.empty((n_trials, n_bins))
    acc_null = np.empty((n_shuffles, n_bins))

    # Cross-validate results
    skf = StratifiedKFold(n_splits=5, shuffle=True)  # Stratified cross-validation

    # Split trials into training and testing sets (each fold gets a unique test set to prevent overfitting)
    for train_index, test_index in skf.split(X, y):

        # Loop over each time bin_train
        for bin_train in range(n_bins):

            # Define train and testing set for the current time bin_train and fold
            X_train, X_test = X[train_index, bin_train], X[test_index, bin_train]
            y_train, y_test = y[train_index], y[test_index]

            # Apply z-scoring normalization across neurons and time bins (otherwise might not converge)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)  # Fit and transform on the training set
            X_test = scaler.transform(X_test)  # Only transform the test set using the same scaler

            # Train decoder (logistic regression) on the current time bin_train’s neural activity
            clf = LogisticRegression()
            clf.fit(X_train, y_train)

            # Evaluate decoder
            y_pred = clf.predict(X_test)  # Predicts the stimulus category for test trials
            y_acc = accuracy_score(y_test, y_pred)  # Computes accuracy for each fold and time bin_train
            # print(f"Accuracy: {y_acc:.2f}")

            # Store results
            pred[test_index, bin_train] = y_pred  # Predicted stimulus condition for each test trial at each time bin_train
            pred_err[test_index, bin_train] = y_pred - y_test  # Difference between predicted and actual labels
            acc[test_index, bin_train] = y_acc  # Accuracy for each test trial at each time bin_train

            # Compute null distribution by shuffling the y_test (faster)
            y_test_shuffled = y_test.values.copy()
            for _ in range(n_shuffles):
                np.random.shuffle(y_test_shuffled)
                acc_null[_, bin_train] = accuracy_score(y_test_shuffled, y_pred)

            # # Compute null distribution by shuffling y_train (slower)
            # y_train_shuffled = y_train.values.copy()
            # for _ in range(n_shuffles):
            #     np.random.shuffle(y_train_shuffled)  # Shuffle independently each iteration
            #     clf.fit(X_train, y_train_shuffled)
            #     y_pred_shuffled = clf.predict(X_test)
            #     acc_null[_, bin_train] = accuracy_score(y_test, y_pred_shuffled)

    return pred, pred_err, acc, acc_null


@timer
def cross_decoder(X=np.zeros((1, 1, 1)), y=np.zeros((1, 1)), n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary stimulus condition from neural data using K-fold
    cross-validation across trials and across time bins.
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary stimulus condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape
    pred = np.empty((n_trials, n_bins, n_bins))
    pred_err = np.empty((n_trials, n_bins, n_bins))
    acc = np.empty((n_trials, n_bins, n_bins))
    acc_null = np.empty((n_shuffles, n_bins, n_bins))

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Cross-validate results
    # kf = KFold()  # K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True)  # Stratified cross-validation

    # Split trials into training and testing sets (each fold gets a unique test set to prevent overfitting)
    # for train_index, test_index in kf.split(X):
    for train_index, test_index in skf.split(X, y):

        # Cross thinghy happens here
        for bin_train in range(X.shape[1]):

            X_train = X[train_index, bin_train]
            y_train = y[train_index]
            # Apply z-scoring normalization to the current time bin_train's data (otherwise might not converge)
            X_train = scaler.fit_transform(X_train)  # Fit and transform on the training set

            # Train decoder (logistic regression) on the current time bin_train’s neural activity
            clf = LogisticRegression()
            clf.fit(X_train, y_train)

            # Loop over each time bin_train
            for bin_test in range(X.shape[1]):

                # Define train and testing set for the current time bin_train and fold
                X_test = X[test_index, bin_test]
                y_test = y[test_index]

                # Apply z-scoring normalization to the current time bin_train's data (otherwise might not converge)
                X_test = scaler.transform(X_test)  # Only transform the test set using the same scaler

                # Evaluate decoder
                y_pred = clf.predict(X_test)  # Predicts the stimulus category for test trials
                y_acc = accuracy_score(y_test, y_pred)  # Computes accuracy for each fold & time bin
                # print(f"Accuracy: {y_acc:.2f}")

                # Store results
                pred[test_index, bin_train, bin_test] = y_pred  # Predicted stimulus condition for each test trial at
                # each time bin
                pred_err[test_index, bin_train, bin_test] = y_pred - y_test  # Difference between predicted and labels
                acc[test_index, bin_train, bin_test] = y_acc  # Accuracy for each test trial at each time bin

                # Compute null distribution by shuffling the labels and evaluating accuracy
                y_test_shuffled = y_test.values.copy()
                for _ in range(n_shuffles):
                    np.random.shuffle(y_test_shuffled)
                    acc_null[_, bin_train, bin_test] = accuracy_score(y_test_shuffled, y_pred)

    return pred, pred_err, acc, acc_null


# Testing getting accuracy per trial instead of per fold
@timer
def epoch_cross_decoder(bins, epoch='stim', X=np.zeros((1, 1, 1)), y=np.zeros((1, 1)), n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary stimulus condition from neural data using K-fold
    cross-validation across trials and across time bins.
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary stimulus condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    n_splits = 5

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape
    pred = np.empty((n_trials, n_bins)) * np.nan
    pred_err = np.empty((n_trials, n_bins)) * np.nan
    # acc = np.empty((n_splits, n_bins))  # Store per fold and bin
    acc = np.empty((n_trials, n_bins)) * np.nan  # Store per trial and bin
    # acc_null = np.empty((n_shuffles, n_bins))  # Store per shuffle and bin
    acc_null = np.empty((n_trials, n_bins, n_shuffles)) * np.nan  # Store per trial, bin, and shuffle
    # pred_err_null = np.empty((X.shape[0], X.shape[1], n_shuffles))

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Cross-validate results
    # kf = KFold()  # K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Stratified cross-validation

    # Define epoch of interest
    if epoch == 'stim':
        epoch_start_idx = np.where(np.round(bins, 1) == 0)[0][0]  # Find index where stim_onset (0) is
        epoch_end_idx = np.where(np.round(bins, 1) == 0.2)[0][0]  # Find index where delay (0.5) is
    elif epoch == 'delay':
        epoch_start_idx = np.where(np.round(bins, 1) == 0.8)[0][0]  # Find index where delay (0.5) is
        epoch_end_idx = np.where(np.round(bins, 1) == 1)[0][0]  # Find index where go cue is in bins
    elif epoch == 'resp':
        epoch_start_idx = np.where(np.round(bins, 1) == 1.8)[0][0]  # Find index where go cue is in bins
        epoch_end_idx = np.where(np.round(bins, 1) == 2)[0][0]  # Find index where go cue is in bins

    # Split trials into training and testing sets (each fold gets a unique test set to prevent overfitting)
    # for train_index, test_index in kf.split(X):
    for k, (train_index, test_index) in enumerate(skf.split(X, y)):

        # Cross thinghy happens here
        X_train = np.mean(X[train_index, epoch_start_idx:epoch_end_idx], axis=1)
        y_train = y[train_index]
        # Apply z-scoring normalization to the current time bin_train's data (otherwise might not converge)
        X_train = scaler.fit_transform(X_train)  # Fit and transform on the training set

        # Train decoder (logistic regression) on the current time bin_train’s neural activity
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        # Loop over each time bin_train
        for bin_test in range(n_bins):

            # Define train and testing set for the current time bin_train and fold
            X_test = X[test_index, bin_test]
            y_test = y[test_index]

            # Apply z-scoring normalization to the current time bin_train's data (otherwise might not converge)
            X_test = scaler.transform(X_test)  # Only transform the test set using the same scaler

            # Evaluate decoder
            y_pred = clf.predict(X_test)  # Predicts the stimulus category for test trials
            # y_acc = accuracy_score(y_test, y_pred)  # Computes accuracy for each fold
            y_acc = (y_pred == y_test).astype(int)  # Computes accuracy per trial
            # print(f"Accuracy: {y_acc:.2f}")

            # Store results
            pred[test_index, bin_test] = y_pred  # Predicted stimulus condition for each test trial at each time
            # bin_train
            pred_err[test_index, bin_test] = y_pred - y_test  # Difference between predicted and actual labels
            # acc[k, bin_test] = y_acc  # Accuracy for each test trial at each time bin
            acc[test_index, bin_test] = y_acc  # Accuracy for each test trial at each time bin

            # Compute null distribution by shuffling the labels and evaluating accuracy
            y_test_shuffled = y_test.values.copy()
            for _ in range(n_shuffles):
                np.random.shuffle(y_test_shuffled)
                # acc_null[_, bin_test] = accuracy_score(y_test_shuffled, y_pred)
                acc_null[test_index, bin_test, _] = (y_pred == y_test_shuffled).astype(int)  # Computes accuracy per trial
                # pred_err_null[test_index, bin_test, _] = y_pred - y_test_shuffled  # Difference between predicted and
                # actual labels

    return pred, pred_err, acc, acc_null#, pred_err_null


@timer
def epoch_cross_decoder_split(bins, split, epoch='stim', X=np.zeros((1, 1, 1)), y=np.zeros((1, 1)), n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary stimulus condition from neural data using K-fold
    cross-validation across trials and across time bins.
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary stimulus condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    n_splits = 5

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape

    pred = [np.empty((n_trials, n_bins)) * np.nan, np.empty((n_trials, n_bins)) * np.nan]
    pred_err = [np.empty((n_trials, n_bins)) * np.nan, np.empty((n_trials, n_bins)) * np.nan]
    acc = [np.empty((n_trials, n_bins)) * np.nan, np.empty((n_trials, n_bins)) * np.nan]
    acc_null = [np.empty((n_trials, n_bins, n_shuffles)) * np.nan, np.empty((n_trials, n_bins, n_shuffles)) * np.nan]

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Cross-validate results
    # kf = KFold()  # K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Stratified cross-validation

    # Define epoch of interest
    if epoch == 'stim':
        epoch_start_idx = np.where(np.round(bins, 1) == 0)[0][0]  # Find index where stim_onset (0) is
        epoch_end_idx = np.where(np.round(bins, 1) == 0.2)[0][0]  # Find index where delay (0.5) is
    elif epoch == 'delay':
        epoch_start_idx = np.where(np.round(bins, 1) == 0.8)[0][0]  # Find index where delay (0.5) is
        epoch_end_idx = np.where(np.round(bins, 1) == 1)[0][0]  # Find index where go cue is in bins
    elif epoch == 'resp':
        epoch_start_idx = np.where(np.round(bins, 1) == 1.8)[0][0]  # Find index where go cue is in bins
        epoch_end_idx = np.where(np.round(bins, 1) == 2)[0][0]  # Find index where go cue is in bins

    # Split trials into training and testing sets (each fold gets a unique test set to prevent overfitting)
    # for train_index, test_index in kf.split(X):
    for k, (train_index, test_index) in enumerate(skf.split(X, y)):

        # Cross thinghy happens here
        X_train = np.mean(X[train_index, epoch_start_idx:epoch_end_idx], axis=1)
        y_train = y[train_index]
        # Apply z-scoring normalization to the current time bin_train's data (otherwise might not converge)
        X_train = scaler.fit_transform(X_train)  # Fit and transform on the training set

        # test_idx0 = np.where(split[test_index] == 0)[0]
        test_idx0 = split.loc[split.index.isin(test_index) & (split == 0)].index
        # test_idx1 = np.where(split[test_index] == 1)[0]
        test_idx1 = split.loc[split.index.isin(test_index) & (split == 1)].index

        # Train decoder (logistic regression) on the current time bin_train’s neural activity
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        # Loop over each time bin_train
        for bin_test in range(n_bins):

            # Define train and testing set for the current time bin_train and fold
            # X_test = X[test_index, bin_test]
            X_test0 = X[test_idx0, bin_test]
            X_test1 = X[test_idx1, bin_test]
            # y_test = y[test_index]
            y_test0 = y[test_idx0]
            y_test1 = y[test_idx1]

            # Apply z-scoring normalization to the current time bin_train's data (otherwise might not converge)
            X_test0 = scaler.transform(X_test0)  # Only transform the test set using the same scaler
            X_test1 = scaler.transform(X_test1)  # Only transform the test set using the same scaler

            # Evaluate decoder
            y_pred0 = clf.predict(X_test0)  # Predicts the stimulus category for test trials
            # y_acc0 = accuracy_score(y_test0, y_pred0)  # Computes accuracy for each fold & time bin_train
            y_acc0 = (y_pred0 == y_test0).astype(int)  # Computes accuracy per trial
            # print(f"Accuracy: {y_acc0:.2f}")
            y_pred1 = clf.predict(X_test1)  # Predicts the stimulus category for test trials
            # y_acc1 = accuracy_score(y_test1, y_pred1)  # Computes accuracy for each fold & time bin_train
            y_acc1 = (y_pred1 == y_test1).astype(int)  # Computes accuracy per trial
            # print(f"Accuracy: {y_acc1:.2f}")

            # Store results
            pred[0][test_idx0, bin_test] = y_pred0  # Predicted stimulus condition for each test trial at each time
            # bin_train
            pred_err[0][test_idx0, bin_test] = y_pred0 - y_test0  # Difference between predicted and actual labels
            # acc0[k, bin_test] = y_acc0  # Accuracy for each test trial at each time bin_train
            acc[0][test_idx0, bin_test] = y_acc0  # Accuracy per trial
            pred[1][test_idx1, bin_test] = y_pred1  # Predicted stimulus condition for each test trial at each time
            # bin_train
            pred_err[1][test_idx1, bin_test] = y_pred1 - y_test1  # Difference between predicted and actual labels
            # acc1[k, bin_test] = y_acc1  # Accuracy for each test trial at each time bin_train
            acc[1][test_idx1, bin_test] = y_acc1  # Accuracy per trial

            # Compute null distribution by shuffling the labels and evaluating accuracy
            y_test_shuffled0 = y_test0.values.copy()
            y_test_shuffled1 = y_test1.values.copy()
            for _ in range(n_shuffles):
                np.random.shuffle(y_test_shuffled0)
                # acc_null0[k, _, bin_test] = accuracy_score(y_test_shuffled0, y_pred0)
                acc_null[0][test_idx0, bin_test, _] = (y_pred0 == y_test_shuffled0).astype(int)  # Accuracy per trial for
                # shuffled labels
                # pred_err_temp.append(y_pred - y_test_shuffled)
                # pred_err_null0[test_idx0, bin_test, _] = y_pred0 - y_test_shuffled0  # Difference between predicted and
                # actual labels

                np.random.shuffle(y_test_shuffled1)
                # acc_null1[k, _, bin_test] = accuracy_score(y_test_shuffled1, y_pred1)
                acc_null[1][test_idx1, bin_test, _] = (y_pred1 == y_test_shuffled1).astype(int)  # Accuracy per trial for
                # shuffled labels
                # pred_err_temp.append(y_pred - y_test_shuffled)
                # pred_err_null1[test_idx1, bin_test, _] = y_pred1 - y_test_shuffled1  # Difference between predicted and
                # actual labels

    return pred, pred_err, acc, acc_null#, pred_err_null0, pred_err_null1


@timer
def mean_decoder(kind=None, epoch=None, split_by=None, save=False, plot=False, engagement=None):
    """
    Perform within time bin decoder across all sessions.
    :param kind: type of decoder (within, cross, epoch)
    :param save: whether to save the results as a pickle
    :param plot: whether to plot the decoding accuracy for each session
    :return: results (dict)
    """

    results = {
        'pred': [],
        'pred_err': [],
        'acc': [],
        'acc_null': [],
        'bins': []
    }

    for i in range(len(ephys_ids)):

        print(f'Processing session {i + 1}/{len(ephys_ids)}: {ephys_ids[i]}...')

        path_behavior = get_behavior_id(ephys_ids[i])
        df_behavior = parse_v2(path_behavior)
        disengagement = find_disengaged(df_behavior, plot=False)  # Find trial when disengagement happens
        engaged = (df_behavior.Trial <= disengagement).astype(int)  # Compare trials to disengagement (no need for i)
        df_behavior['Engaged'] = engaged
        bins = np.load(folder_parent / ephys_ids[i] / 'bins.npy')
        all_psth = np.load(folder_parent / ephys_ids[i] / 'all_psth.npy')

        # Engagement
        if engagement == 0:  # Disengaged trials
            df_behavior = df_behavior[df_behavior.Engaged == 0].reset_index(drop=True)
        elif engagement == 1:  # Engaged trials
            df_behavior = df_behavior[df_behavior.Engaged == 1].reset_index(drop=True)
        all_psth = all_psth[df_behavior.index.values]

        # Misses
        resp_idx = df_behavior[df_behavior.Miss == 0].index
        # resp_idx = df_behavior[df_behavior.Hit == 1].index
        df_behavior = df_behavior.iloc[resp_idx].reset_index(drop=True)
        all_psth = all_psth[resp_idx]
        # split = df_behavior[split_by]

        if kind == 'within':
            pred, pred_err, acc, acc_null = within_decoder(all_psth, df_behavior.Side)
            filename = 'results_mean_within_decoder.pkl'
        elif kind == 'cross':
            pred, pred_err, acc, acc_null = cross_decoder(all_psth, df_behavior.Side)
            filename = 'results_mean_cross_decoder.pkl'
        elif kind == 'epoch':
            pred, pred_err, acc, acc_null = epoch_cross_decoder(bins, epoch, all_psth, df_behavior.Side)
            filename = 'results_mean_epoch_cross_decoder' + '_' + epoch + '.pkl'
        elif kind == 'epoch_split':
            split = df_behavior[split_by]
            pred, pred_err, acc, acc_null = epoch_cross_decoder_split(bins, split, epoch=epoch, X=all_psth,
                                                                      y=df_behavior.Side)
            filename = 'results_mean_epoch_cross_decoder' + '_' + epoch + '_' + 'split_by' + '_' + split_by + '.pkl'
        elif kind == 'test':
            pass
            filename = 'results_mean_TEST.pkl'

        results['pred'].append(pred)
        results['pred_err'].append(pred_err)
        results['acc'].append(acc)
        results['acc_null'].append(acc_null)
        results['bins'] = bins

    if save:
        os.chdir(folder_parent)
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

    # Plot decoding accuracy for each session
    if plot:
        if kind == 'within':
            plot_within_decoder(bins, results['acc'][i], results['acc_null'][i], z_null=True)
        elif kind == 'cross':
            plot_cross_decoder(bins, acc, acc_null, z_null=True)
        elif kind == 'epoch':
            pass
        elif kind == 'epoch_split':
            pass

    return results


def null_zscore(acc, acc_null):
    """
    Compute and z-scores for the decoding accuracy relative to the null distribution.
    :param acc: 2D array with decoding accuracy (trials x time)
    :param acc_null: 3D array with null distribution of accuracy (shuffles x time)
    :return: z_scores
    """

    # Compute p-values and z-scores
    acc_mean = np.mean(acc, axis=0)  # Mean accuracy across trials
    null_acc_mean = np.mean(acc_null, axis=0)  # Mean accuracy across shuffles
    null_acc_std = np.std(acc_null, axis=0)  # Standard deviation of accuracy across shuffles
    null_acc_std[null_acc_std < 0.1] = 0.1  # Avoid division by zero
    z_scores = ((acc_mean - null_acc_mean) / null_acc_std)  # Z-score for each time bin

    return z_scores


def p_val(acc, acc_null):
    """
    Compute p-values for the decoding accuracy relative to the null distribution.
    :param acc: 2D array with decoding accuracy (trials x time)
    :param acc_null: 3D array with null distribution of accuracy (shuffles x time)
    :return: p_values
    """

    acc_mean = np.mean(acc, axis=0)  # Mean accuracy across trials
    p_values = np.mean(acc_null > acc_mean,
                       axis=0)  # p-value as the fraction of shuffles where null accuracy > real accuracy
    return p_values


########################################################################################################################
# PLOT DECODERS
########################################################################################################################

def plot_within_decoder(bins, acc, acc_null, z_null=True):
    """
    Plot the decoding accuracy and the null distribution of accuracy.
    :param bins: 1D array with time bins
    :param acc: 2D array with decoding accuracy (trials x time)
    :param acc_null: 2D array with null distribution of accuracy (shuffles x time)
    :param z_null: whether to Z-score the decoding accuracy by the null distribution of accuracy
    """

    # Plot decoding accuracy
    plt.figure(constrained_layout=True)

    # Compute p-values and z-scores relative to the null distribution
    if z_null:
        z_scores = null_zscore(acc, acc_null)
        p_values = p_val(acc, acc_null)
        significant_region = p_values < 0.05  # When assessing significance of single sessions use p < 0.05
        plt.plot(bins[:-1], z_scores, label='Z acc.')
        plt.fill_between(bins[:-1], z_scores, where=significant_region, edgecolor='none',
                         alpha=0.25, label='α < .05')
        plt.axhline(0, color='tab:gray', linestyle='--')  # Chance level
        ylabel = 'Z-score'
    else:
        acc_mean = acc.mean(axis=0)
        acc_sem = sem(acc, axis=0)
        acc_null_mean = acc_null.mean(axis=0)
        acc_null_band = np.percentile(acc_null, [2.5, 97.5], axis=0)  # The 95% confidence interval of the shuffles
        # plt.plot(-np.mean(abs(pred_err), axis=0)+1)  # Equivalent
        plt.plot(bins[:-1], acc_mean, label='Acc.')
        plt.fill_between(bins[:-1], acc_mean - acc_sem, acc_mean + acc_sem, edgecolor='none', alpha=0.25,
                         label='Acc. s.e.m.')
        plt.plot(bins[:-1], acc_null_mean, color='tab:gray', linestyle='-', label='Null mean')  # Chance level (0.5)
        plt.fill_between(bins[:-1], acc_null_band[0], acc_null_band[1], color='tab:gray', edgecolor='none', alpha=0.25,
                         label='Null 95% CI')
        ylabel = 'Accuracy'

    plt.axvline(0, color='tab:red', linestyle='-')  # Stimulus onset
    plt.axvline(1, color='tab:gray', linestyle='-')  # Go cue
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(f'Decoding accuracy\n'
              f'{df_behavior.Subject.unique()[0]}, {acc.shape[0]} trials')
    plt.legend(frameon=False)
    sns.despine()


def plot_cross_decoder(bins, acc, acc_null, z_null=True):
    """
    Plot the prediction error for the cross-temporal decoder.
    :param bins: 1D array with time bins
    :param acc: 3D array with decoding accuracy (trials x time x time)
    :param acc_null: 3D array with null distribution of accuracy (shuffles x time x time)
    :param z_null: whether to Z-score the decoding accuracy by the null distribution of accuracy
    """
    plt.figure(constrained_layout=True)

    if z_null:
        z_score = null_zscore(acc, acc_null)
        plt.imshow(z_score, origin='lower', cmap='RdBu_r', norm=CenteredNorm())  # abs needed?
        axislabel = ' - Z-score'
    else:
        plt.imshow(np.mean(acc, axis=0), origin='lower')  # abs needed?
        axislabel = ''

    plt.colorbar()
    plt.xticks(np.arange(0, len(bins), 10), np.round(bins[::10]).astype(int))
    plt.yticks(np.arange(0, len(bins), 10), np.round(bins[::10]).astype(int))
    plt.axhline(np.where(bins == 0)[0], color='k', linestyle='-')  # Stimulus onset
    plt.axvline(np.where(bins == 0)[0], color='k', linestyle='-')  # Stimulus onset
    plt.axhline(np.where(bins == 0.5)[0], color='k', linestyle='-')  # Delay
    plt.axvline(np.where(bins == 0.5)[0], color='k', linestyle='-')  # Delay
    plt.axhline(np.where(bins == 1)[0], color='k', linestyle='-')  # Go cue
    plt.axvline(np.where(bins == 1)[0], color='k', linestyle='-')  # Go cue
    plt.xlabel('Test time (s)' + axislabel)
    plt.ylabel('Train time (s)' + axislabel)
    # plt.title('Cross temporal decoder')
    # plt.title(f'Decoding accuracy\n'
    #           f'{df_behavior.Subject.unique()[0]}, {acc.shape[0]} trials')
    sns.despine()


def plot_epoch_cross_decoder(bins, acc, acc_null, epoch='stim', z_null=True):

    """
    Plot the epoch cross temporal decoding accuracy of a single session.
    :param bins: 1D array with time bins
    :param acc: 2D array with decoding accuracy (folds x time bins)
    :param acc_null: 2D array with null distribution of accuracy (shuffles x time bins)
    :param epoch: epoch of interest (stim, delay, resp)
    :param z_null: whether to Z-score the decoding accuracy by the null distribution of accuracy
    """

    if epoch == 'stim':
        label = 'Stimulus'
    elif epoch == 'delay':
        label = 'Delay'
    elif epoch == 'resp':
        label = 'Response'

    acc_mean = np.mean(acc, axis=0)
    acc_sem = sem(acc, axis=0)
    acc_null_mean = np.mean(acc_null, axis=(0, 2))
    acc_null_sem = sem(acc_null, axis=(0, 2))
    n_trials = len(acc)

    plt.figure(constrained_layout=True)
    plt.plot(bins[:-1], acc_mean, label=label)
    plt.fill_between(bins[:-1], acc_mean - acc_sem, acc_mean + acc_sem, edgecolor='none', alpha=0.25)
    plt.plot(bins[:-1], acc_null_mean, color='tab:gray', linestyle='--', label='Null')
    plt.fill_between(bins[:-1], acc_null_mean - acc_null_sem, acc_null_mean + acc_null_sem, edgecolor='none',
                     color='tab:gray', alpha=0.25)
    plt.legend(frameon=False)
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.title(f'Decoding accuracy\n'
              f' {n_trials} trials')
    sns.despine()


def plot_mean_within_decoder(results, z_null=True):
    """
    Plot the mean decoding accuracy across all sessions.
    :param results: dict with decoding results for each session
    :return:
    """

    plt.figure(constrained_layout=True)

    # Compute the mean accuracy across all sessions
    acc_mean = [results['acc'][i].mean(axis=0) for i in range(len(results['acc']))]
    acc_mean = np.array(acc_mean)
    acc_null_mean = [results['acc_null'][i].mean(axis=0) for i in range(len(results['acc_null']))]
    acc_null_mean = np.array(acc_null_mean)
    n_trials = np.sum([results['acc'][i].shape[0] for i in range(len(results['acc']))])
    bins = results['bins']

    if z_null:
        z_scores = [null_zscore(results['acc'][i], results['acc_null'][i]) for i in range(len(results['acc']))]
        z_scores_mean = np.mean((z_scores), axis=0)
        plt.plot(bins[:-1], z_scores_mean, color='tab:blue', label='Z acc.')
        p_values = p_val(acc_mean, acc_null_mean)
        # significant_region = p_values < 0.05  # When assessing significance across sessions use p < 0.05
        significant_region = np.abs(z_scores_mean) >= 1.96  # When assessing significance across sessions use 1.96
        plt.fill_between(bins[:-1], z_scores_mean, where=significant_region, edgecolor='none', color='tab:blue',
                         alpha=0.25, label='α < .05')
        plt.axhline(0, color='tab:gray', linestyle='--')  # Chance level
        ylabel = 'Z-score'

    else:
        # Plot the mean decoding accuracy across all sessions
        plt.plot(bins[:-1], np.mean(acc_mean, axis=0), color='tab:blue', label='Acc.')
        plt.fill_between(bins[:-1], np.mean(acc_mean, axis=0) - sem(acc_mean, axis=0),
                         np.mean(acc_mean, axis=0) + sem(acc_mean, axis=0), color='tab:blue', edgecolor='none',
                         alpha=0.25, label='Acc. s.e.m.')

        # Plot the mean null accuracy across all sessions (chance level)
        plt.plot(bins[:-1], np.mean(acc_null_mean, axis=0), color='tab:gray', label='Acc. null')
        plt.fill_between(bins[:-1], np.mean(acc_null_mean, axis=0) - sem(acc_null_mean, axis=0),
                         np.mean(acc_null_mean, axis=0) + sem(acc_null_mean, axis=0), color='tab:gray',
                         edgecolor='none', alpha=0.25, label='Acc. null s.e.m.')

        # Plot the individual sessions null accuracy (chance level)
        # for _ in range(len(results['acc_null'])):
        #     plt.plot(bins[:-1], np.mean(results['acc_null'][_], axis=0), ls='--', c='tab:gray')
        ylabel = 'Accuracy'

    plt.axvline(0, color='tab:red', linestyle='-')  # Stimulus onset
    # plt.axvline(go_cue, color='tab:gray', linestyle='-')  # Go cue
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    # plt.title(f"Decoding accuracy\n"
    #           f"{subject}, {len(results['acc'])} sessions, {n_trials} trials")
    sns.despine()
    plt.legend(frameon=False)


def plot_mean_cross_decoder(results, z_null=True):
    """
    Plot the mean decoding accuracy across all sessions.
    :param results: dict with decoding results for each session
    :return:
    """

    plt.figure(constrained_layout=True)

    # Compute the mean accuracy across all sessions
    acc_mean = [results['acc'][i].mean(axis=0) for i in range(len(results['acc']))]
    acc_mean = np.array(acc_mean)
    acc_mean = np.mean(acc_mean, axis=0)
    # acc_null_mean = [results['acc_null'][i].mean(axis=0) for i in range(len(results['acc_null']))]
    # acc_null_mean = np.array(acc_null_mean)
    # acc_null_mean = np.mean(acc_null_mean, axis=0)
    n_trials = np.sum([results['acc'][i].shape[0] for i in range(len(results['acc']))])
    bins = results['bins']

    if z_null:
        z_scores = [null_zscore(results['acc'][i], results['acc_null'][i]) for i in range(len(results['acc']))]
        z_scores_mean = np.mean((z_scores), axis=0)
        plt.imshow(z_scores_mean, origin='lower', cmap='RdBu_r', norm=CenteredNorm())
    else:
        plt.imshow(acc_mean, origin='lower')  # abs needed?

    plt.colorbar()
    plt.xticks(np.arange(0, len(bins), 10), np.round(bins[::10]).astype(int))
    plt.yticks(np.arange(0, len(bins), 10), np.round(bins[::10]).astype(int))
    plt.axhline(np.where(bins == 0)[0], color='k', linestyle='-')  # Stimulus onset
    plt.axvline(np.where(bins == 0)[0], color='k', linestyle='-')  # Stimulus onset
    plt.axhline(np.where(bins == 0.5)[0], color='k', linestyle='-')  # Delay
    plt.axvline(np.where(bins == 0.5)[0], color='k', linestyle='-')  # Delay
    plt.axhline(np.where(bins == 1)[0], color='k', linestyle='-')  # Go cue
    plt.axvline(np.where(bins == 1)[0], color='k', linestyle='-')  # Go cue
    plt.xlabel('Test time (s)')
    plt.ylabel('Train time (s)')
    # plt.title(f"Decoding accuracy\n"
    #           f" {subject}, {len(results['acc'])} sessions, {n_trials} trials")
    sns.despine()


def plot_mean_epoch_cross_decoder(results, epoch='stim', z_null=True):
    """
    Plot the mean epoch cross temporal decoding accuracy across all sessions.
    :param results: dict with decoding results for each session
    :param z_null:
    """

    if epoch == 'stim':
        label = 'Stimulus'
    elif epoch == 'delay':
        label = 'Delay'
    elif epoch == 'resp':
        label = 'Response'

    acc_mean = [np.nanmean(results['acc'][i], axis=0) for i in range(len(results['acc']))]
    acc_mean = np.array(acc_mean)
    acc_sem = sem(acc_mean, axis=0, nan_policy='omit')
    acc_mean = np.nanmean(acc_mean, axis=0)
    # n_trials = np.sum([results['pred'][i].shape[0] for i in range(len(results['acc']))])
    # plt.figure(constrained_layout=True)
    plt.plot(results['bins'][:-1], acc_mean, label=label)
    plt.fill_between(results['bins'][:-1], acc_mean - acc_sem, acc_mean + acc_sem, edgecolor='none', alpha=0.25)
    plt.legend(frameon=False)
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    # plt.title(f'Decoding accuracy\n'
    #           f'{subject}, {len(results["acc"])} sessions, {n_trials} trials')
    sns.despine()


def plot_mean_epoch_cross_decoder_split(results, z_null=True):
    """
    Plot the mean epoch cross temporal decoding accuracy across all sessions.
    :param results: dict with decoding results for each session
    :param z_null: whether to Z-score the decoding accuracy by the null distribution of accuracy
    """

    # Plot mean accuracy epoch stimulus decoder split by Hit
    acc0_mean = np.nanmean([np.nanmean(results['acc'][i][0], axis=0) for i in range(len(results['acc']))], axis=0)
    acc0_sem = sem([np.nanmean(results['acc'][i][0], axis=0) for i in range(len(results['acc']))], axis=0)
    acc_null0_mean = np.nanmean(
        [(np.nanmean(results['acc_null'][i][0], axis=(0, 2))) for i in range(len(results['acc_null']))], axis=0)

    acc1_mean = np.nanmean([np.nanmean(results['acc'][i][1], axis=0) for i in range(len(results['acc']))], axis=0)
    acc1_sem = sem([np.nanmean(results['acc'][i][1], axis=0) for i in range(len(results['acc']))], axis=0)
    acc_null1_mean = np.nanmean(
        [(np.nanmean(results['acc_null'][i][1], axis=(0, 2))) for i in range(len(results['acc_null']))], axis=0)

    plt.figure(constrained_layout=True)

    plt.plot(results['bins'][:-1], acc0_mean, color='tab:red', label='Error')
    plt.fill_between(results['bins'][:-1], acc0_mean - acc0_sem, acc0_mean + acc0_sem, color='tab:red',
                     edgecolor='none', alpha=0.25)
    plt.plot(results['bins'][:-1], acc_null0_mean, color='tab:red', alpha=0.5, linestyle='--')

    plt.plot(results['bins'][:-1], acc1_mean, color='tab:green', label='Correct')
    plt.fill_between(results['bins'][:-1], acc1_mean - acc1_sem, acc1_mean + acc1_sem, color='tab:green',
                     edgecolor='none', alpha=0.25)
    plt.plot(results['bins'][:-1], acc_null1_mean, color='tab:green', alpha=0.5, linestyle='--')

    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.legend(frameon=False)
    sns.despine()


########################################################################################################################
# UNDER CONSTRUCTION
# Please wear safety equipment. Any resemblance to real persons, living or dead, is purely coincidental. I do not take
# responsibility for any damage caused by the use of this code. Use at your own risk.
########################################################################################################################

# Testing split according to engagement
@timer
def epoch_cross_decoder_TEST(bins, epoch='stim', X=np.zeros((1, 1, 1)), y=np.zeros((1, 1)), eng=np.zeros((1, 1)),
                             n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary stimulus condition from neural data using K-fold
    cross-validation across trials and across time bins.
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary stimulus condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    # Get the indexes of the trials where the animal was engaged
    engaged = eng[eng == 1].index.values
    disengaged = eng[eng == 0].index.values

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape
    pred = np.empty((n_trials, n_bins)) * np.nan
    pred_err = np.empty((n_trials, n_bins)) * np.nan
    acc = np.empty((n_trials, n_bins)) * np.nan  # Store per trial and bin
    acc_null = np.empty((n_trials, n_bins, n_shuffles)) * np.nan  # Store per trial, bin, and shuffle

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Define epoch of interest
    if epoch == 'stim':
        epoch_start_idx = np.where(np.round(bins, 1) == 0)[0][0]  # Find index where stim_onset (0) is
        epoch_end_idx = np.where(np.round(bins, 1) == 0.2)[0][0]  # Find index where delay (0.5) is
    elif epoch == 'delay':
        epoch_start_idx = np.where(np.round(bins, 1) == 0.8)[0][0]  # Find index where delay (0.5) is
        epoch_end_idx = np.where(np.round(bins, 1) == 1)[0][0]  # Find index where go cue is in bins
    elif epoch == 'resp':
        epoch_start_idx = np.where(np.round(bins, 1) == 1.8)[0][0]  # Find index where go cue is in bins
        epoch_end_idx = np.where(np.round(bins, 1) == 2)[0][0]  # Find index where go cue is in bins

    # Cross thinghy happens here
    # Train decoder on engaged trials
    X_train = np.mean(X[engaged, epoch_start_idx:epoch_end_idx], axis=1)
    X_train = scaler.fit_transform(X_train)  # Apply z-scoring normalization to the training and test sets
    y_train = y[engaged]

    # # Test on disengaged trials
    # X_test = np.mean(X[disengaged, epoch_start_idx:epoch_end_idx], axis=1)
    # y_test = y[disengaged]

    # Train decoder (logistic regression) on the current time bin neural activity
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Loop over each time bin_train
    for bin_test in range(n_bins):

        # Define test set for the current time bin
        X_test = X[disengaged, bin_test]
        y_test = y[disengaged]

        # Apply z-scoring normalization to the current time bin_train's data (otherwise might not converge)
        X_test = scaler.transform(X_test)  # Only transform the test set using the same scaler

        # Evaluate decoder
        y_pred = clf.predict(X_test)  # Predicts the stimulus category for test trials
        y_acc = (y_pred == y_test).astype(int)  # Computes accuracy per trial

        # Store results
        pred[disengaged, bin_test] = y_pred  # Predicted stimulus condition for each test trial at each time bin
        pred_err[disengaged, bin_test] = y_pred - y_test  # Difference between predicted and actual labels
        acc[disengaged, bin_test] = y_acc  # Accuracy for each test trial at each time bin

        # Compute null distribution by shuffling the labels and evaluating accuracy
        y_test_shuffled = y_test.values.copy()
        for _ in range(n_shuffles):
            np.random.shuffle(y_test_shuffled)
            acc_null[disengaged, bin_test, _] = (y_pred == y_test_shuffled).astype(int)  # Computes accuracy per trial

    return pred, pred_err, acc, acc_null
