import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
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

ephys_ids = [
    '007_2024-06-22_10-48-57',
    '007_2024-06-23_12-46-55',
    '007_2024-06-24_17-47-22',
    '007_2024-06-27_15-06-28',
    '007_2024-07-09_12-10-57',
    '007_2024-07-10_12-03-35',
    '007_2024-07-11_12-39-21',
    '007_2024-07-12_13-29-26'
]

behavior_ids = [
    '007_stage_training_v5_20240622-110354',
    '007_stage_training_v5_20240623-130152',
    '007_stage_training_v5_20240624-180217',
    '007_stage_training_v5_20240627-152129',
    '007_stage_training_v5_20240709-122550',
    '007_stage_training_v5_20240710-121827',
    '007_stage_training_v5_20240711-125439',
    '007_stage_training_v5_20240712-134450'
]

subject = '007'

folder_parent = Path.home() / 'data' / subject

for i in range(len(ephys_ids)):

    print(f'Processing session {i + 1}/{len(ephys_ids)}: {ephys_ids[i]}...')

    # Create child folder within parent folder for each ephys_id with its name if it doesn't exist
    folder_child = folder_parent / ephys_ids[i]
    folder_child.mkdir(parents=True, exist_ok=True)
    os.chdir(folder_child)

    # Execute only if folder is empty
    if len(os.listdir(folder_child)) > 0:
        print('Folder is not empty. Skipping...')
        continue
    else:
        print('Folder is empty. Proceeding...')
        df_ttl, df_behavior, n_trials, df_spikes, cluster_info, x, height, labels, y, width, left, ts_edges, events_edges = \
            preprocess(ephys_ids[i])
        bins, all_psth = get_all_psth(cluster_info, df_spikes, df_ttl, n_trials, time_win=[-1, 3], bin_size=0.1)

    # Save bins and psth
    np.save(folder_child / 'bins.npy', bins)
    np.save(folder_child / 'all_psth.npy', all_psth)


# Load behavioral data from all sessions in a list of DataFrames
behavior = []
for i in range(len(behavior_ids)):
    path_behavior = (Path.home() / 'Downloads' / behavior_ids[i]).with_suffix('.csv')
    df_behavior = parse_v2(path_behavior)
    behavior.append(df_behavior)

# Get behavioral events
stim_dur = df_behavior.StimDur.unique()[0]
delay = df_behavior.Delay.unique()[0]
go_cue = stim_dur + delay

########################################################################################################################

# Define functions for decoding
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
    pred = np.empty((X.shape[0], X.shape[1]))
    pred_err = np.empty((X.shape[0], X.shape[1]))
    acc = np.empty((X.shape[0], X.shape[1]))
    acc_null = np.empty((n_shuffles, X.shape[1]))

    # Cross-validate results
    # kf = KFold()  # K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True)  # Stratified cross-validation

    # Split trials into training and testing sets (each fold gets a unique test set to prevent overfitting)
    # for train_index, test_index in kf.split(X):
    for train_index, test_index in skf.split(X, y):

        # Loop over each time bin_train
        for bin_train in range(X.shape[1]):

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
            y_acc = accuracy_score(y_test, y_pred)  # Computes accuracy for each fold & time bin_train
            # print(f"Accuracy: {y_acc:.2f}")

            # Store results
            pred[test_index, bin_train] = y_pred  # Predicted stimulus condition for each test trial at each time bin
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
    pred = np.empty((X.shape[0], X.shape[1], X.shape[1]))
    pred_err = np.empty((X.shape[0], X.shape[1], X.shape[1]))
    acc = np.empty((X.shape[0], X.shape[1], X.shape[1]))
    acc_null = np.empty((n_shuffles, X.shape[1], X.shape[1]))

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
                y_acc = accuracy_score(y_test, y_pred)  # Computes accuracy for each fold & time bin_train
                # print(f"Accuracy: {y_acc:.2f}")

                # Store results
                pred[
                    test_index, bin_train, bin_test] = y_pred  # Predicted stimulus condition for each test trial at each time bin_train
                pred_err[
                    test_index, bin_train, bin_test] = y_pred - y_test  # Difference between predicted and actual labels
                acc[test_index, bin_train, bin_test] = y_acc  # Accuracy for each test trial at each time bin_train

                # Compute null distribution by shuffling the labels and evaluating accuracy
                y_test_shuffled = y_test.values.copy()
                for _ in range(n_shuffles):
                    np.random.shuffle(y_test_shuffled)
                    acc_null[_, bin_train, bin_test] = accuracy_score(y_test_shuffled, y_pred)

    return pred, pred_err, acc, acc_null


@timer
def epoch_cross_decoder(bins, epoch='stim', X=np.zeros((1, 1, 1)), y=np.zeros((1, 1)), n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary stimulus condition from neural data using K-fold
    cross-validation across trials and across time bins.
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary stimulus condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    # Initialize arrays
    pred = np.empty((X.shape[0], X.shape[1]))
    pred_err = np.empty((X.shape[0], X.shape[1]))
    acc = np.empty((X.shape[0], X.shape[1]))
    acc_null = np.empty((n_shuffles, X.shape[1]))
    pred_err_null = np.empty((X.shape[0], X.shape[1], n_shuffles))

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Cross-validate results
    # kf = KFold()  # K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True)  # Stratified cross-validation

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
    for train_index, test_index in skf.split(X, y):

        # Cross thinghy happens here
        X_train = np.mean(X[train_index, epoch_start_idx:epoch_end_idx], axis=1)
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
            y_acc = accuracy_score(y_test, y_pred)  # Computes accuracy for each fold & time bin_train
            # print(f"Accuracy: {y_acc:.2f}")

            # Store results
            pred[test_index, bin_test] = y_pred  # Predicted stimulus condition for each test trial at each time
            # bin_train
            pred_err[test_index, bin_test] = y_pred - y_test  # Difference between predicted and actual labels
            acc[test_index, bin_test] = y_acc  # Accuracy for each test trial at each time bin_train

            # Compute null distribution by shuffling the labels and evaluating accuracy
            y_test_shuffled = y_test.values.copy()
            pred_err_temp = []
            for _ in range(n_shuffles):
                np.random.shuffle(y_test_shuffled)
                acc_null[_, bin_test] = accuracy_score(y_test_shuffled, y_pred)
                # pred_err_temp.append(y_pred - y_test_shuffled)
                pred_err_null[test_index, bin_test, _] = y_pred - y_test_shuffled  # Difference between predicted and
                # actual labels

    return pred, pred_err, acc, acc_null, pred_err_null












@timer
def epoch_cross_decoder_split(hit, bins, epoch='stim', X=np.zeros((1, 1, 1)), y=np.zeros((1, 1)), n_shuffles=10):
    """
    Perform logistic regression-based decoding of a binary stimulus condition from neural data using K-fold
    cross-validation across trials and across time bins.
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary stimulus condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    n_splits = 5

    len_idx0 = len(np.where(hit == 0)[0])
    len_idx1 = len(np.where(hit == 1)[0])

    # Initialize arrays

    # indexes[0]
    pred0 = np.empty((len_idx0, X.shape[1]))
    pred_err0 = np.empty((len_idx0, X.shape[1]))
    acc0 = np.empty((n_splits, X.shape[1]))
    acc_null0 = np.empty((n_splits, n_shuffles, X.shape[1]))
    pred_err_null0 = np.empty((len_idx0, X.shape[1], n_shuffles))

    # indexes[1]
    pred1 = np.empty((len_idx1, X.shape[1]))
    pred_err1 = np.empty((len_idx1, X.shape[1]))
    acc1 = np.empty((n_splits, X.shape[1]))
    acc_null1 = np.empty((n_splits, n_shuffles, X.shape[1]))
    pred_err_null1 = np.empty((len_idx1, X.shape[1], n_shuffles))

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Cross-validate results
    # kf = KFold()  # K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True)  # Stratified cross-validation

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

        test_idx0 = np.where(hit[test_index] == 0)[0]
        test_idx1 = np.where(hit[test_index] == 1)[0]

        # Train decoder (logistic regression) on the current time bin_train’s neural activity
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        # Loop over each time bin_train
        for bin_test in range(X.shape[1]):

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
            y_acc0 = accuracy_score(y_test0, y_pred0)  # Computes accuracy for each fold & time bin_train
            # print(f"Accuracy: {y_acc0:.2f}")
            y_pred1 = clf.predict(X_test1)  # Predicts the stimulus category for test trials
            y_acc1 = accuracy_score(y_test1, y_pred1)  # Computes accuracy for each fold & time bin_train
            # print(f"Accuracy: {y_acc1:.2f}")

            # Store results
            pred0[test_idx0, bin_test] = y_pred0  # Predicted stimulus condition for each test trial at each time
            # bin_train
            pred_err0[test_idx0, bin_test] = y_pred0 - y_test0  # Difference between predicted and actual labels
            acc0[k, bin_test] = y_acc0  # Accuracy for each test trial at each time bin_train
            pred1[test_idx1, bin_test] = y_pred1  # Predicted stimulus condition for each test trial at each time
            # bin_train
            pred_err1[test_idx1, bin_test] = y_pred1 - y_test1  # Difference between predicted and actual labels
            acc1[k, bin_test] = y_acc1  # Accuracy for each test trial at each time bin_train


            # Compute null distribution by shuffling the labels and evaluating accuracy
            y_test_shuffled0 = y_test0.values.copy()
            y_test_shuffled1 = y_test1.values.copy()
            for _ in range(n_shuffles):
                np.random.shuffle(y_test_shuffled0)
                acc_null0[k, _, bin_test] = accuracy_score(y_test_shuffled0, y_pred0)
                # pred_err_temp.append(y_pred - y_test_shuffled)
                pred_err_null0[test_idx0, bin_test, _] = y_pred0 - y_test_shuffled0   # Difference between predicted and
                # actual labels

                np.random.shuffle(y_test_shuffled1)
                acc_null1[k, _, bin_test] = accuracy_score(y_test_shuffled1, y_pred1)
                # pred_err_temp.append(y_pred - y_test_shuffled)
                pred_err_null1[test_idx1, bin_test, _] = y_pred1 - y_test_shuffled1   # Difference between predicted and
                # actual labels

    return pred0, pred1, pred_err0, pred_err1, acc0, acc1, acc_null0, acc_null1, pred_err_null0, pred_err_null1


pred0, pred1, pred_err0, pred_err1, acc0, acc1, acc_null0, acc_null1, pred_err_null0, pred_err_null1 = \
    epoch_cross_decoder_split(hit, bins, epoch='stim', X=all_psth, y=df_behavior.Side, n_shuffles=10)


plt.figure(constrained_layout=True)
plt.plot(bins[:-1], np.mean(acc0, axis=0), color='tab:red', label='Error')
plt.fill_between(bins[:-1], np.mean(acc0, axis=0) - sem(acc0, axis=0), np.mean(acc0, axis=0) + sem(acc0, axis=0),
                    color='tab:red', alpha=0.25)
plt.plot(bins[:-1], np.mean(acc1, axis=0), color='tab:green', label='Correct')
plt.fill_between(bins[:-1], np.mean(acc1, axis=0) - sem(acc1, axis=0), np.mean(acc1, axis=0) + sem(acc1, axis=0),
                    color='tab:green', alpha=0.25)









# pred, pred_err, acc, acc_null = within_decoder(X=all_psth, y=df_behavior.Side, n_shuffles=1000)
# pred, pred_err, acc, acc_null = cross_decoder(X=all_psth, y=df_behavior.Side, n_shuffles=100)


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
    plt.axvline(go_cue, color='tab:gray', linestyle='-')  # Go cue
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
    plt.xticks(np.arange(0, len(bins), 10), np.round(bins[::10]))
    plt.yticks(np.arange(0, len(bins), 10), np.round(bins[::10]))
    plt.axhline(np.where(bins == 0)[0], color='k', linestyle='-')  # Stimulus onset
    plt.axvline(np.where(bins == 0)[0], color='k', linestyle='-')  # Stimulus onset
    plt.axhline(np.where(bins == go_cue)[0], color='k', linestyle='-')  # Go cue
    plt.axvline(np.where(bins == go_cue)[0], color='k', linestyle='-')  # Go cue
    plt.xlabel('Test time (s)' + axislabel)
    plt.ylabel('Train time (s)' + axislabel)
    # plt.title('Cross temporal decoder')
    plt.title(f'Decoding accuracy\n'
              f'{df_behavior.Subject.unique()[0]}, {acc.shape[0]} trials')
    sns.despine()


@timer
def mean_within_decoder(plot=False):
    """
    Perform within time bin decoder across all sessions.
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
        path_behavior = (Path.home() / 'Downloads' / behavior_ids[i]).with_suffix('.csv')
        # df_ttl, df_behavior, n_trials, df_spikes, cluster_info, x, height, labels, y, width, left, ts_edges, events_edges = \
        #     preprocess(ephys_ids[i], path_behavior)
        # bins, all_psth = get_all_psth(cluster_info, df_spikes, df_ttl, n_trials, time_win=[-1, 3], bin_size=0.1)
        df_behavior = parse_v2(path_behavior)
        bins = np.load(folder_parent / ephys_ids[i] / 'bins.npy')
        all_psth = np.load(folder_parent / ephys_ids[i] / 'all_psth.npy')
        pred, pred_err, acc, acc_null = within_decoder(all_psth, df_behavior.Side)

        results['pred'].append(pred)
        results['pred_err'].append(pred_err)
        results['acc'].append(acc)
        results['acc_null'].append(acc_null)
        results['bins'] = bins

        # Plot decoding accuracy for each session
        if plot:
            plot_within_decoder(bins, results['acc'][i], results['acc_null'][i], z_null=True)

    return results


@timer
def mean_cross_decoder(save=False, plot=False):
    """
    Perform within time bin decoder across all sessions.
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
        path_behavior = (Path.home() / 'Downloads' / behavior_ids[i]).with_suffix('.csv')
        # df_ttl, df_behavior, n_trials, df_spikes, cluster_info, x, height, labels, y, width, left, ts_edges, events_edges = \
        #     preprocess(ephys_ids[i], path_behavior)
        # bins, all_psth = get_all_psth(cluster_info, df_spikes, df_ttl, n_trials, time_win=[-1, 3], bin_size=0.1)
        df_behavior = parse_v2(path_behavior)
        bins = np.load(folder_parent / ephys_ids[i] / 'bins.npy')
        all_psth = np.load(folder_parent / ephys_ids[i] / 'all_psth.npy')
        pred, pred_err, acc, acc_null = cross_decoder(all_psth, df_behavior.Side)

        results['pred'].append(pred)
        results['pred_err'].append(pred_err)
        results['acc'].append(acc)
        results['acc_null'].append(acc_null)
        results['bins'] = bins

        if save:
            os.chdir(folder_parent)
            with open('results_mean_cross_decoder.pkl', 'wb') as f:
                pickle.dump(results, f)

        # Plot decoding accuracy for each session
        if plot:
            plot_within_decoder(bins, results['acc'][i], results['acc_null'][i], z_null=True)

    return results


with open('results_mean_cross_decoder.pkl', 'rb') as f:
    data_loaded = pickle.load(f)


@timer
def mean_epoch_cross_decoder(epoch='stim', plot=False):
    """
    Perform within time bin decoder across all sessions.
    :param plot: whether to plot the decoding accuracy for each session
    :return: results (dict)
    """

    results = {
        'pred': [],
        'pred_err': [],
        'acc': [],
        'acc_null': [],
        'pred_err_null': [],
        'bins': []
    }

    for i in range(len(ephys_ids)):
        print(f'Processing session {i + 1}/{len(ephys_ids)}: {ephys_ids[i]}...')
        path_behavior = (Path.home() / 'Downloads' / behavior_ids[i]).with_suffix('.csv')
        # df_ttl, df_behavior, n_trials, df_spikes, cluster_info, x, height, labels, y, width, left, ts_edges, events_edges = \
        #     preprocess(ephys_ids[i], path_behavior)
        # bins, all_psth = get_all_psth(cluster_info, df_spikes, df_ttl, n_trials, time_win=[-1, 3], bin_size=0.1)
        df_behavior = parse_v2(path_behavior)
        bins = np.load(folder_parent / ephys_ids[i] / 'bins.npy')
        all_psth = np.load(folder_parent / ephys_ids[i] / 'all_psth.npy')

        pred, pred_err, acc, acc_null, pred_err_null = epoch_cross_decoder(bins, epoch, all_psth, df_behavior.Side)
        results['pred'].append(pred)
        results['pred_err'].append(pred_err)
        results['acc'].append(acc)
        results['acc_null'].append(acc_null)
        results['pred_err_null'].append(pred_err_null)
        results['bins'] = bins

        # # Plot decoding accuracy for each session
        # if plot:
        #     plot_within_decoder(bins, results['acc'][i], results['acc_null'][i], z_null=True)

    return results










def mean_epoch_cross_decoder_split(epoch='stim', plot=False):
    """
    Perform within time bin decoder across all sessions.
    :param plot: whether to plot the decoding accuracy for each session
    :return: results (dict)
    """

    results = {

        'pred0': [],
        'pred1': [],
        'pred_err0': [],
        'pred_err1': [],
        'acc0': [],
        'acc1': [],
        'acc_null0': [],
        'acc_null1': [],
        'pred_err_null0': [],
        'pred_err_null1': []
    }

    for i in range(len(ephys_ids)):
        print(f'Processing session {i + 1}/{len(ephys_ids)}: {ephys_ids[i]}...')
        path_behavior = (Path.home() / 'Downloads' / behavior_ids[i]).with_suffix('.csv')
        df_ttl, df_behavior, n_trials, df_spikes, cluster_info, x, height, labels, y, width, left, ts_edges, events_edges = \
            preprocess(ephys_ids[i], path_behavior)
        bins, all_psth = get_all_psth(cluster_info, df_spikes, df_ttl, n_trials, time_win=[-1, 3], bin_size=0.1)
        # df_behavior = parse_v2(path_behavior)
        # bins = np.load(folder_parent / ephys_ids[i] / 'bins.npy')
        # all_psth = np.load(folder_parent / ephys_ids[i] / 'all_psth.npy')

        pred0, pred1, pred_err0, pred_err1, acc0, acc1, acc_null0, acc_null1, pred_err_null0, pred_err_null1 = \
            epoch_cross_decoder_split(bins, epoch, all_psth, df_behavior.Side)
        results['pred0'].append(pred0)
        results['pred_err0'].append(pred_err0)
        results['acc0'].append(acc0)
        results['acc_null0'].append(acc_null0)
        results['pred_err_null0'].append(pred_err_null0)
        results['pred1'].append(pred1)
        results['pred_err1'].append(pred_err1)
        results['acc1'].append(acc1)
        results['acc_null1'].append(acc_null1)
        results['pred_err_null1'].append(pred_err_null1)
        results['bins'] = bins

        # # Plot decoding accuracy for each session
        # if plot:
        #     plot_within_decoder(bins, results['acc'][i], results['acc_null'][i], z_null=True)

    return results

results = mean_epoch_cross_decoder_split(epoch='stim', plot=False)








results_stim = mean_epoch_cross_decoder(epoch='stim', plot=False)
results_delay = mean_epoch_cross_decoder(epoch='delay', plot=False)
results_resp = mean_epoch_cross_decoder(epoch='resp', plot=False)
results = {
    'result_stim': results_stim,
    'results_delay': results_delay,
    'results_resp': results_resp
}

# Save results as pickle
os.chdir(folder_parent)
with open('results_mean_epoch_cross_decoder.pkl', 'wb') as f:
    pickle.dump(results, f)





acc_mean_stim = [results_stim['acc'][i].mean(axis=0) for i in range(len(results_stim['acc']))]
acc_mean_stim = np.array(acc_mean_stim)
acc_mean_delay = [results_delay['acc'][i].mean(axis=0) for i in range(len(results_delay['acc']))]
acc_mean_delay = np.array(acc_mean_delay)
acc_mean_resp = [results_resp['acc'][i].mean(axis=0) for i in range(len(results_resp['acc']))]
acc_mean_resp = np.array(acc_mean_resp)
n_trials = np.sum([results_stim['acc'][i].shape[0] for i in range(len(results_stim['acc']))])

plt.figure(constrained_layout=True)
plt.plot(results_stim['bins'][:-1], np.mean(acc_mean_stim, axis=0), label='Stimulus')
plt.fill_between(results_stim['bins'][:-1], np.mean(acc_mean_stim, axis=0) - sem(acc_mean_stim, axis=0),
                    np.mean(acc_mean_stim, axis=0) + sem(acc_mean_stim, axis=0), edgecolor='none', alpha=0.25)
plt.plot(results_delay['bins'][:-1], np.mean(acc_mean_delay, axis=0), label='Delay')
plt.fill_between(results_delay['bins'][:-1], np.mean(acc_mean_delay, axis=0) - sem(acc_mean_delay, axis=0),
                    np.mean(acc_mean_delay, axis=0) + sem(acc_mean_delay, axis=0), edgecolor='none', alpha=0.25)
plt.plot(results_resp['bins'][:-1], np.mean(acc_mean_resp, axis=0), label='Response')
plt.fill_between(results_resp['bins'][:-1], np.mean(acc_mean_resp, axis=0) - sem(acc_mean_resp, axis=0),
                    np.mean(acc_mean_resp, axis=0) + sem(acc_mean_resp, axis=0), edgecolor='none', alpha=0.25)
plt.legend(frameon=False)
plt.xlabel('Time (s)')
plt.ylabel('Accuracy')
plt.title(f'Decoding accuracy\n'
          f'{subject}, {len(results_stim["acc"])} sessions, {n_trials} trials')
sns.despine()

# results = mean_within_decoder(plot=False)
# results = mean_cross_decoder(plot=False)


# for i in range(len(ephys_ids)):
#     plot_within_decoder(bins, results['acc'][i], results['acc_null'][i], z_null=True)


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
        plt.plot(bins[:-1], z_scores_mean, label='Z acc.')
        # p_values = p_val(acc_mean, acc_null_mean)
        # significant_region = p_values < 0.05  # When assessing significance across sessions use p < 0.05
        significant_region = np.abs(z_scores_mean) >= 1.96  # When assessing significance across sessions use 1.96
        plt.fill_between(bins[:-1], z_scores_mean, where=significant_region, edgecolor='none',
                         alpha=0.25, label='α < .05')
        plt.axhline(0, color='tab:gray', linestyle='--')  # Chance level
        ylabel = 'Z-score'

    else:
        # Plot the mean decoding accuracy across all sessions
        plt.plot(bins[:-1], np.mean(acc_mean, axis=0), label='Acc.')
        plt.fill_between(bins[:-1], np.mean(acc_mean, axis=0) - sem(acc_mean, axis=0),
                         np.mean(acc_mean, axis=0) + sem(acc_mean, axis=0), edgecolor='none', alpha=0.25,
                         label='Acc. s.e.m.')
        # Plot the mean null accuracy across all sessions (chance level)
        for _ in range(len(results['acc_null'])):
            plt.plot(bins[:-1], np.mean(results['acc_null'][_], axis=0), ls='--', c='tab:gray')
        ylabel = 'Accuracy'

    plt.axvline(0, color='tab:red', linestyle='-')  # Stimulus onset
    plt.axvline(go_cue, color='tab:gray', linestyle='-')  # Go cue
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(f"Decoding accuracy\n"
              f"{subject}, {len(results['acc'])} sessions, {n_trials} trials")
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
    plt.xticks(np.arange(0, len(bins), 10), np.round(bins[::10]))
    plt.yticks(np.arange(0, len(bins), 10), np.round(bins[::10]))
    plt.axhline(np.where(bins == 0)[0], color='k', linestyle='-')  # Stimulus onset
    plt.axvline(np.where(bins == 0)[0], color='k', linestyle='-')  # Stimulus onset
    plt.axhline(np.where(bins == delay)[0], color='k', linestyle='-')  # Delay
    plt.axvline(np.where(bins == delay)[0], color='k', linestyle='-')  # Delay
    plt.axhline(np.where(bins == go_cue)[0], color='k', linestyle='-')  # Go cue
    plt.axvline(np.where(bins == go_cue)[0], color='k', linestyle='-')  # Go cue
    plt.xlabel('Test time (s)')
    plt.ylabel('Train time (s)')
    # plt.title('Cross temporal decoder')
    # plt.title(f"Decoding accuracy\n"
    #           f"{df_behavior.Subject.unique()[0]}, {len(results['acc'])} trials")
    plt.title(f"Decoding accuracy\n"
              f" 007, {len(results['acc'])} sessions, {n_trials} trials")
    sns.despine()


# Split stim decoder into correct vs error trials
indexes = [get_trial_indexes(behavior[i], condition='outcome') for i in range(len(behavior))]

# Correct trials
mae_stim_correct = [-np.abs(results_stim['pred_err'][i][indexes[i][1]]).mean(axis=0) + 1 for i in range(len(results_stim['pred_err']))]
mae_stim_correct = np.array(mae_stim_correct)
avg_pred_err_null = [results_stim['pred_err_null'][i].mean(axis=2) for i in range(len(results_stim['pred_err_null']))]
mae_stim_correct_null = [-np.abs(avg_pred_err_null[i][indexes[i][1]]).mean(axis=0)+1 for i in range(len(results_stim['pred_err_null']))]
mae_stim_correct_null = np.array(mae_stim_correct_null)

# Error trials
mae_stim_error = [-np.abs(results_stim['pred_err'][i][indexes[i][0]]).mean(axis=0) + 1 for i in range(len(results_stim['pred_err']))]
mae_stim_error = np.array(mae_stim_error)
avg_pred_err_null = [results_stim['pred_err_null'][i].mean(axis=2) for i in range(len(results_stim['pred_err_null']))]
mae_stim_error_null = [-np.abs(avg_pred_err_null[i][indexes[i][0]]).mean(axis=0)+1 for i in range(len(results_stim['pred_err_null']))]
mae_stim_error_null = np.array(mae_stim_error_null)

plt.figure(constrained_layout=True)
color = 'tab:green'
plt.plot(results_stim['bins'][:-1], mae_stim_correct.mean(axis=0), color=color, label='Correct')
plt.fill_between(results_stim['bins'][:-1], mae_stim_correct.mean(axis=0) - sem(mae_stim_correct, axis=0),
                 mae_stim_correct.mean(axis=0) + sem(mae_stim_correct, axis=0), color=color, edgecolor='none',
                 alpha=0.25)
color = 'tab:red'
plt.plot(results_stim['bins'][:-1], mae_stim_error.mean(axis=0), color=color, label='Error')
plt.fill_between(results_stim['bins'][:-1], mae_stim_error.mean(axis=0) - sem(mae_stim_error, axis=0),
                 mae_stim_error.mean(axis=0) + sem(mae_stim_error, axis=0), color=color, edgecolor='none',
                 alpha=0.25)
plt.legend(frameon=False)





