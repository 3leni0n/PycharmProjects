import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import matplotlib.patches as patches
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
import pickle
import traceback
from joblib import Parallel, delayed

# Neuromatch tutorial: https://compneuro.neuromatch.io/tutorials/W1D5_DeepLearning/student/W1D5_Tutorial1.html

########################################################################################################################

def preprocess_subject(subject, align='stim', time_win=[-1, 3], bin_size=0.1):
    """
    Preprocess all ephys and behavioral sessions for a given subject and save the results in a folder.
    :param subject: subject ID (str)
    :return: None
    """

    folder_parent = Path.home() / 'data' / subject
    ephys_ids = get_ephys_sessions(subject)
    error_sessions = []

    # Compute spike counts for all neurons of a session (bins, all_psth). If file doesn't exist, create it
    for i in range(len(ephys_ids)):
        print(f'Processing session {i + 1}/{len(ephys_ids)}: {ephys_ids[i]}')

        # Create child folder within parent folder for each ephys_id with its name if it doesn't exist
        folder_child = folder_parent / ephys_ids[i]
        folder_child.mkdir(parents=True, exist_ok=True)
        os.chdir(folder_child)

        # .npy files (spike counts)
        if any(f.endswith('.npy') for f in os.listdir(folder_child)):
            print("'.npy' files exist in folder. Skipping...")
        else:
            try:
                print("'.npy' files do not exist in folder. Proceeding...")
                preprocessed = preprocess(ephys_ids[i])
                df_ttl, df_behavior, n_trials, df_spikes, cluster_info, x, height, labels, y, width, left, ts_edges, events_edges = tuple(preprocessed)
                bins, all_psth = get_all_psth(cluster_info, df_spikes, df_ttl, n_trials, align=align, time_win=time_win,
                                              bin_size=bin_size)
                np.save(folder_child / f'bins_{align}.npy', bins)
                np.save(folder_child / f'all_psth_{align}.npy', all_psth)

            except Exception as e:
                print(f'An error occurred: {e}')
                traceback.print_exc()
                error_sessions.append(ephys_ids[i])

        # .csv files (behavior)
        if any(f.endswith('.csv') for f in os.listdir(folder_child)):
            print("'.csv' files exist in folder. Skipping...")
        else:
            try:
                print("'.csv' files do not exist in folder. Proceeding...")
                experiment = df_behavior.Experiment.unique()[0]
                filename = df_behavior.Session.unique()[0] + '.csv'
                df_behavior.to_csv(folder_child / filename, index=False)
            except Exception as e:
                print(f'An error occurred: {e}')
                traceback.print_exc()
                error_sessions.append(ephys_ids[i])

    # # Save the glmhmm results if they don't exist
    # # Requite probably to fit the GLMHMM to all data acquisition sessions (not only recorded ones)
    # path_glmhmm =  Path.home() / 'PycharmProjects' / 'glmhmm' / experiment / f'{subject}.csv'
    # df_glmhmm = pd.read_csv(path_glmhmm)

    print(f'Sessions not preprocessed: {error_sessions}')

    return error_sessions


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

    print(f'Disengagement in trial {disengaged_trial}')

    # Side accuracy plot
    if plot:
        plt.figure(constrained_layout=True)
        plt.plot(x_total, y_total, color='k', label='Total')
        plt.plot(x_0, y_0, color='tab:blue', label='Left')
        plt.plot(x_1, y_1, color='tab:orange', label='Right')
        plt.axhline(0.25, color='tab:gray', ls=':')
        plt.axhline(0.5, color='tab:gray', ls='--')
        plt.axhline(0.75, color='tab:gray', ls=':')

        # Plot the red dot at the correct x (absolute trial number) and y=0.5
        if disengaged_trial is not None:
            plt.plot(disengaged_trial, threshold, 'ro')

        plt.xlabel('Trial')
        plt.ylabel('Accuracy')
        plt.title(df_behavior.Session.unique()[0])
        plt.legend(frameon=False)
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
    Perform logistic regression-based decoding of a binary condition from neural data using K-fold cross-validation.
    Train and test within the same time bin.
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape
    pred = np.empty((n_trials, n_bins))
    pred_err = np.empty((n_trials, n_bins))
    acc = np.empty((n_trials, n_bins))
    acc_null = np.empty((n_shuffles, n_bins))

    # Cross-validate results
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Stratified cross-validation

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
            clf = LogisticRegression(class_weight='balanced')
            clf.fit(X_train, y_train)

            # Evaluate decoder
            y_pred = clf.predict(X_test)  # Predicts the stimulus category for test trials
            # y_acc = accuracy_score(y_test, y_pred)  # Computes accuracy for each fold and time bin_train
            y_acc = (y_pred == y_test).astype(int)  # Accuracy per trial
            # print(f"Accuracy: {y_acc:.2f}")

            # Store results
            pred[test_index, bin_train] = y_pred  # Predicted stimulus condition for each test trial at each time bin_train
            pred_err[test_index, bin_train] = y_pred - y_test  # Difference between predicted and actual labels
            acc[test_index, bin_train] = y_acc  # Accuracy for each test trial at each time bin_train

            # Compute null distribution by shuffling the y_test (faster)
            y_test_shuffled = y_test.values.copy()
            for _ in range(n_shuffles):
                np.random.shuffle(y_test_shuffled)
                # acc_null[_, bin_train] = accuracy_score(y_test_shuffled, y_pred)
                acc_null[_, bin_train] = np.mean((y_pred == y_test_shuffled).astype(int))

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
    Perform logistic regression-based decoding of a binary condition from neural data using K-fold cross-validation.
    Train in one bin and test in the rest, for each time bin.
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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Stratified cross-validation

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
            clf = LogisticRegression(class_weight='balanced')
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
def epoch_cross_decoder(bins, epoch=None, X=np.zeros((1, 1, 1)), y=np.zeros((1, 1)), n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary condition from neural data using K-fold cross-validation.
    Train in one bin and test in the rest, for one epoch (one bin or the mean of a few). Akin of takin a slice of the
    cross-decoder matrix.
    :param bins: 1D array with time bins
    :param epoch: epoch of interest (str: 'stim', 'delay', 'resp')
    :param X: 3D array with neural data (trials x time x neurons)
    :param Y: 1D array with binary stimulus condition
    :param n_shuffles: number of shuffles to perform
    :return: pred, pred_err (predicted condition and prediction error)
    """

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape
    pred = np.empty((n_trials, n_bins)) * np.nan
    pred_err = np.empty((n_trials, n_bins)) * np.nan
    # acc = np.empty((n_splits, n_bins))  # Store per fold and bin
    acc = np.empty((n_trials, n_bins)) * np.nan  # Store per trial and bin
    # acc_null = np.empty((n_shuffles, n_bins))  # Store per shuffle and bin
    acc_null = np.empty((n_trials, n_bins, n_shuffles)) * np.nan  # Store per trial, bin, and shuffle

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Cross-validate results
    # kf = KFold()  # K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Stratified cross-validation

    # Define epoch of interest

    # Align to stimulus onset
    if epoch == 'stim':
        epoch_start_idx = np.where(np.round(bins, 1) == 0)[0][0]  # Find index where stim_onset (0) is
        epoch_end_idx = np.where(np.round(bins, 1) == 0.1)[0][0]  # Find index where delay (0.5) is
    elif epoch == 'delay':
        epoch_start_idx = np.where(np.round(bins, 1) == 0.9)[0][0]  # Find index where delay (0.5) is
        epoch_end_idx = np.where(np.round(bins, 1) == 1)[0][0]  # Find index where go cue is in bins
    elif epoch == 'resp':
        epoch_start_idx = np.where(np.round(bins, 1) == 1.9)[0][0]  # Find index where go cue is in bins
        epoch_end_idx = np.where(np.round(bins, 1) == 2)[0][0]  # Find index where go cue is in bins

    # Align to first lick
    elif epoch == 'first_lick':
        epoch_start_idx = np.where(np.round(bins, 2) == -0.05)[0][0]  # Find index where first lick is in bins
        epoch_end_idx = np.where(np.round(bins, 2) == 0)[0][0]  # Find index where first lick is in bins
    elif epoch == 'mid_lick':
        epoch_start_idx = np.where(np.round(bins, 2) == 0.5)[0][0]  # Find index where first lick is in bins
        epoch_end_idx = np.where(np.round(bins, 2) == 0.55)[0][0]  # Find index where first lick is in bins
    else:
        raise ValueError("Epoch must be 'stim', 'delay', or 'resp'.")

    # Split trials into training and testing sets (each fold gets a unique test set to prevent overfitting)
    # for train_index, test_index in kf.split(X):
    for k, (train_index, test_index) in enumerate(skf.split(X, y)):

        # Cross thinghy happens here
        X_train = np.mean(X[train_index, epoch_start_idx:epoch_end_idx], axis=1)
        y_train = y[train_index]
        # Apply z-scoring normalization to the current time bin_train's data (otherwise might not converge)
        X_train = scaler.fit_transform(X_train)  # Fit and transform on the training set

        # Train decoder (logistic regression) on the current time bin_train’s neural activity
        clf = LogisticRegression(class_weight='balanced')
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

    return pred, pred_err, acc, acc_null


@timer
def epoch_cross_decoder_split(bins, split, epoch=None, X=np.zeros((1, 1, 1)), y=np.zeros((1, 1)), n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary condition from neural data using K-fold cross-validation.
    Train in one bin and test in the rest, for one epoch (one bin or the mean of a few). Akin of takin a slice of the
    cross-decoder matrix.
    Train in all trials and test separately for each condition.
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape
    pred = [np.empty((n_trials, n_bins)) * np.nan, np.empty((n_trials, n_bins)) * np.nan]
    pred_err = [np.empty((n_trials, n_bins)) * np.nan, np.empty((n_trials, n_bins)) * np.nan]
    acc = [np.empty((n_trials, n_bins)) * np.nan, np.empty((n_trials, n_bins)) * np.nan]
    acc_null = [np.empty((n_trials, n_bins, n_shuffles)) * np.nan, np.empty((n_trials, n_bins, n_shuffles)) * np.nan]

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Cross-validate results
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define epoch of interest
    if epoch == 'stim':  # Find index where stim_onset (0-0.5) is
        epoch_start_idx = np.where(np.round(bins, 1) == 0)[0][0]
        epoch_end_idx = np.where(np.round(bins, 1) == 0.1)[0][0]
    elif epoch == 'delay':  # Find index where delay (0.5-1) is
        epoch_start_idx = np.where(np.round(bins, 1) == 0.9)[0][0]
        epoch_end_idx = np.where(np.round(bins, 1) == 1)[0][0]
    elif epoch == 'resp':  # Find index where go cue (1-2) is
        epoch_start_idx = np.where(np.round(bins, 1) == 1.9)[0][0]
        epoch_end_idx = np.where(np.round(bins, 1) == 2)[0][0]

    # Split trials into training and testing sets (each fold gets a unique test set to prevent over-fitting)
    for k, (train_index, test_index) in enumerate(skf.split(X, y)):

        # Cross thingy happens here
        X_train = np.mean(X[train_index, epoch_start_idx:epoch_end_idx], axis=1)
        y_train = y[train_index]
        X_train = scaler.fit_transform(X_train)  # Fit and transform on the training set

        test_idx0 = split.loc[split.index.isin(test_index) & (split == 0)].index
        test_idx1 = split.loc[split.index.isin(test_index) & (split == 1)].index

        # Train decoder (logistic regression) on the current time bin’s neural activity
        clf = LogisticRegression(class_weight='balanced')  # Handle imbalanced classes (bias)
        clf.fit(X_train, y_train)

        # Loop over each time bin_train
        for bin_test in range(n_bins):

            # Define train and testing set for the current time bin_train and fold
            X_test0 = X[test_idx0, bin_test]
            X_test1 = X[test_idx1, bin_test]
            y_test0 = y[test_idx0]
            y_test1 = y[test_idx1]

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
            pred[0][test_idx0, bin_test] = y_pred0  # Predicted stimulus condition for each test trial at each time bin
            pred_err[0][test_idx0, bin_test] = y_pred0 - y_test0  # Difference between predicted and actual labels
            # acc0[k, bin_test] = y_acc0  # Accuracy for each test trial at each time bin
            acc[0][test_idx0, bin_test] = y_acc0  # Accuracy per trial
            pred[1][test_idx1, bin_test] = y_pred1  # Predicted stimulus condition for each test trial at each time bin
            pred_err[1][test_idx1, bin_test] = y_pred1 - y_test1  # Difference between predicted and actual labels
            # acc1[k, bin_test] = y_acc1  # Accuracy for each test trial at each time bin
            acc[1][test_idx1, bin_test] = y_acc1  # Accuracy per trial

            # Compute null distribution by shuffling the labels and evaluating accuracy
            y_test_shuffled0 = y_test0.values.copy()
            y_test_shuffled1 = y_test1.values.copy()
            for _ in range(n_shuffles):
                # np.random.shuffle(y_test_shuffled0)
                # # acc_null0[k, _, bin_test] = accuracy_score(y_test_shuffled0, y_pred0)
                # acc_null[0][test_idx0, bin_test, _] = (y_pred0 == y_test_shuffled0).astype(int)  # Accuracy per trial for
                # # shuffled labels
                # # pred_err_temp.append(y_pred - y_test_shuffled)
                # # pred_err_null0[test_idx0, bin_test, _] = y_pred0 - y_test_shuffled0  # Difference between predicted and
                # # actual labels
                #
                # np.random.shuffle(y_test_shuffled1)
                # # acc_null1[k, _, bin_test] = accuracy_score(y_test_shuffled1, y_pred1)
                # acc_null[1][test_idx1, bin_test, _] = (y_pred1 == y_test_shuffled1).astype(int)  # Accuracy per trial for
                # # shuffled labels
                # # pred_err_temp.append(y_pred - y_test_shuffled)
                # # pred_err_null1[test_idx1, bin_test, _] = y_pred1 - y_test_shuffled1  # Difference between predicted and
                # # actual labels

                # TEST FOR NON FLAT SHUFFLES
                y_train_shuffled = np.random.permutation(y_train)
                clf_null = LogisticRegression(class_weight='balanced')
                clf_null.fit(X_train, y_train_shuffled)

                # Predict on the same test sets
                y_pred0_null = clf_null.predict(X_test0)
                y_pred1_null = clf_null.predict(X_test1)

                # Compute per-trial accuracy
                acc_null[0][test_idx0, bin_test, _] = (y_pred0_null == y_test0).astype(int)
                acc_null[1][test_idx1, bin_test, _] = (y_pred1_null == y_test1).astype(int)

    return pred, pred_err, acc, acc_null


@timer
def epoch_cross_decoder_generalize(bins, split, epoch=None, X=np.zeros((1, 1, 1)), y=np.zeros((1, 1)), n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary condition from neural data using K-fold cross-validation.
    Train in one bin and test in the rest, for one epoch (one bin or the mean of a few). Akin of takin a slice of the
    cross-decoder matrix.
    Train in one condition and test in that condition (cross-validation) and in the other condition (generalization).
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape
    pred = [np.empty((n_trials, n_bins)) * np.nan, np.empty((n_trials, n_bins)) * np.nan]
    pred_err = [np.empty((n_trials, n_bins)) * np.nan, np.empty((n_trials, n_bins)) * np.nan]
    acc = [np.empty((n_trials, n_bins)) * np.nan, np.empty((n_trials, n_bins)) * np.nan]
    acc_null = [np.empty((n_trials, n_bins, n_shuffles)) * np.nan, np.empty((n_trials, n_bins, n_shuffles)) * np.nan]

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Cross-validate results
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define epoch of interest
    if epoch == 'stim':  # Find index where stim_onset (0-0.5) is
        epoch_start_idx = np.where(np.round(bins, 1) == 0)[0][0]
        epoch_end_idx = np.where(np.round(bins, 1) == 0.1)[0][0]
    elif epoch == 'delay':  # Find index where delay (0.5-1) is
        epoch_start_idx = np.where(np.round(bins, 1) == 0.9)[0][0]
        epoch_end_idx = np.where(np.round(bins, 1) == 1)[0][0]
    elif epoch == 'resp':  # Find index where go cue (1-2) is
        epoch_start_idx = np.where(np.round(bins, 1) == 1.9)[0][0]
        epoch_end_idx = np.where(np.round(bins, 1) == 2)[0][0]

    index1 = np.where(split == 1)[0]  # Indices of correct/engaged trials
    index0 = np.where(split == 0)[0]  # Indices of error/disengaged trials (never used for training)

    # Split trials into training and testing sets (each fold gets a unique test set to prevent over-fitting)
    for k, (train_index, test_index) in enumerate(skf.split(X[index1], y[index1])):

        # Cross thingy happens here
        train_index = index1[train_index]  # Map indices back to full dataset
        test_idx1 = index1[test_index]  # Correct/engaged trials for testing
        test_idx0 = index0  # Use all error/disengaged trials for testing

        X_train = np.mean(X[train_index, epoch_start_idx:epoch_end_idx], axis=1)
        y_train = y[train_index]
        X_train = scaler.fit_transform(X_train)  # Fit and transform on the training set

        # test_idx0 = split.loc[split.index.isin(test_index) & (split == 0)].index
        # test_idx1 = split.loc[split.index.isin(test_index) & (split == 1)].index

        # Train decoder (logistic regression) on the current time bin’s neural activity
        clf = LogisticRegression(class_weight='balanced')  # Handle imbalanced classes (bias)
        clf.fit(X_train, y_train)

        # Loop over each time bin_train
        for bin_test in range(n_bins):

            # Define train and testing set for the current time bin_train and fold
            X_test0 = X[test_idx0, bin_test]
            X_test1 = X[test_idx1, bin_test]
            y_test0 = y[test_idx0]
            y_test1 = y[test_idx1]

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
            pred[0][test_idx0, bin_test] = y_pred0  # Predicted stimulus condition for each test trial at each time bin
            pred_err[0][test_idx0, bin_test] = y_pred0 - y_test0  # Difference between predicted and actual labels
            # acc0[k, bin_test] = y_acc0  # Accuracy for each test trial at each time bin
            acc[0][test_idx0, bin_test] = y_acc0  # Accuracy per trial
            pred[1][test_idx1, bin_test] = y_pred1  # Predicted stimulus condition for each test trial at each time bin
            pred_err[1][test_idx1, bin_test] = y_pred1 - y_test1  # Difference between predicted and actual labels
            # acc1[k, bin_test] = y_acc1  # Accuracy for each test trial at each time bin
            acc[1][test_idx1, bin_test] = y_acc1  # Accuracy per trial

            # Compute null distribution by shuffling the labels and evaluating accuracy
            y_test_shuffled0 = y_test0.values.copy()
            y_test_shuffled1 = y_test1.values.copy()
            for _ in range(n_shuffles):
                # np.random.shuffle(y_test_shuffled0)
                # # acc_null0[k, _, bin_test] = accuracy_score(y_test_shuffled0, y_pred0)
                # acc_null[0][test_idx0, bin_test, _] = (y_pred0 == y_test_shuffled0).astype(int)  # Accuracy per trial for
                # # shuffled labels
                # # pred_err_temp.append(y_pred - y_test_shuffled)
                # # pred_err_null0[test_idx0, bin_test, _] = y_pred0 - y_test_shuffled0  # Difference between predicted and
                # # actual labels
                #
                # np.random.shuffle(y_test_shuffled1)
                # # acc_null1[k, _, bin_test] = accuracy_score(y_test_shuffled1, y_pred1)
                # acc_null[1][test_idx1, bin_test, _] = (y_pred1 == y_test_shuffled1).astype(int)  # Accuracy per trial for
                # # shuffled labels
                # # pred_err_temp.append(y_pred - y_test_shuffled)
                # # pred_err_null1[test_idx1, bin_test, _] = y_pred1 - y_test_shuffled1  # Difference between predicted and
                # # actual labels

                # TEST FOR NON FLAT SHUFFLES
                y_train_shuffled = np.random.permutation(y_train)
                clf_null = LogisticRegression(class_weight='balanced')
                clf_null.fit(X_train, y_train_shuffled)

                # Predict on the same test sets
                y_pred0_null = clf_null.predict(X_test0)
                y_pred1_null = clf_null.predict(X_test1)

                # Compute per-trial accuracy
                acc_null[0][test_idx0, bin_test, _] = (y_pred0_null == y_test0).astype(int)
                acc_null[1][test_idx1, bin_test, _] = (y_pred1_null == y_test1).astype(int)

    return pred, pred_err, acc, acc_null


@timer
def mean_decoder(subject, what='stim', align='stim', kind=None, epoch=None, epoch_ortho=None, split_by=None, drop_miss=True,
                 hit_only=False, engagement=None, n_shuffles=100, plot=False, save=False):
    """
    Perform within time bin decoder across all sessions for one subject.
    :param subject: subject ID (str)
    :param what: what to decode ('stim' or 'choice'). If correct trials only, they are the same ('choice'='stim')
    :param align: alignment of neural data ('stim', 'go_cue', 'resp')
    :param kind: type of decoder (within, cross, epoch)
    :param epoch: epoch of interest (str: 'stim', 'delay', 'resp')
    :param epoch_ortho: epoch to orthogonalize against (str: 'stim', 'delay', 'resp')
    :param split_by: column to split trials by (str)
    :param drop_miss: whether to drop miss trials (default: True)
    :param hit_only: whether to use only hit trials (default: False)
    :param engagement: whether to filter by engagement (1: engaged, 0: disengaged, None: all trials)
    :param n_shuffles: number of shuffles to perform
    :param plot: whether to plot the decoding accuracy for each session
    :param save: whether to save the results as a pickle
    :return: results (dict)
    """

    if what == 'stim':
        col = 'Side'
    elif what == 'choice':
        col = 'Choice'
    else:
        raise ValueError("'what' (to decode) must be 'stim' or 'choice'.")

    results = {
        'pred': [],
        'pred_err': [],
        'acc': [],
        'acc_null': [],
        'bins': []
    }

    folder_parent = Path.home() / 'data' / subject
    ephys_ids = get_ephys_sessions(subject)
    # error_sessions = preprocess_subject(subject)
    # ephys_ids = [id for id in ephys_ids if id not in error_sessions]

    error_sessions = []

    for i in range(len(ephys_ids)):

        folder_child = folder_parent / ephys_ids[i]

        try:
            print(f'Processing session {i + 1}/{len(ephys_ids)}: {ephys_ids[i]}...')
            # path_behavior = get_behavior_id(ephys_ids[i])
            # df_behavior = parse_v2(path_behavior)
            filename_behavior = [f for f in os.listdir(folder_child) if f.endswith('.csv')][0]  # Assume only one .csv file
            path_behavior = folder_child / filename_behavior
            df_behavior = pd.read_csv(path_behavior)
            bins = np.load(folder_child / f'bins_{align}.npy')
            all_psth = np.load(folder_child / f'all_psth_{align}.npy')

            state_label = ''
            if engagement is not None:  # Add engaged column to df_behavior
            # if engagement is None:  # Add engaged column to df_behavior
                disengagement = find_disengaged(df_behavior, plot=False)  # Find trial where disengagement happens
                engaged = (df_behavior.Trial <= disengagement).astype(int)  # Label trials as engaged/disengaged
                df_behavior['Engaged'] = engaged
                if engagement == 0:  # Disengaged trials
                    df_behavior = df_behavior[df_behavior.Engaged == 0].reset_index(drop=True)
                    state_label = '_disengaged'
                elif engagement == 1:  # Engaged trials
                    df_behavior = df_behavior[df_behavior.Engaged == 1].reset_index(drop=True)
                    state_label = '_engaged'
                all_psth = all_psth[df_behavior.index.values]  # Filter by engagement
            # else:
            #     state_label = ''

            # Filter trials
            # Remove misses (default)
            if drop_miss:
                resp_idx = df_behavior[df_behavior.Miss == 0].index
                df_behavior = df_behavior.iloc[resp_idx].reset_index(drop=True)
                all_psth = all_psth[resp_idx]

            # Correct trials only (not default)
            if hit_only:
                correct_idx = df_behavior[df_behavior.Hit == 1].index
                df_behavior = df_behavior.iloc[correct_idx].reset_index(drop=True)
                all_psth = all_psth[correct_idx]
                hit_label = '_hit_only'
            else:
                hit_label = ''

            # Select decoder
            if kind == 'within':
                pred, pred_err, acc, acc_null = \
                    within_decoder(all_psth, df_behavior[col], n_shuffles)
                filename = f'{what}_{kind}_decoder{hit_label}{state_label}_({align} aligned).pkl'
            elif kind == 'cross':
                pred, pred_err, acc, acc_null = \
                    cross_decoder(all_psth, df_behavior[col], n_shuffles)
                filename = f'{what}_{kind}{hit_label}{state_label}_({align} aligned).pkl'
            elif kind == 'epoch':
                pred, pred_err, acc, acc_null = \
                    epoch_cross_decoder(bins, epoch, all_psth, df_behavior[col], n_shuffles)
                filename = f'{what}_{kind}_decoder_{epoch}{hit_label}{state_label}_({align} aligned).pkl'
            elif kind == 'epoch_split':
                split = df_behavior[split_by]
                pred, pred_err, acc, acc_null = \
                    epoch_cross_decoder_split(bins, split, epoch, all_psth, df_behavior[col], n_shuffles)
                filename = f'{what}_{kind}_decoder_{epoch}_split_by_{split_by}{hit_label}{state_label}_({align} aligned).pkl'
            elif kind == 'epoch_generalize':
                split = df_behavior[split_by]
                pred, pred_err, acc, acc_null = \
                    epoch_cross_decoder_generalize(bins, split, epoch, all_psth, df_behavior[col], n_shuffles)
                filename = f'{what}_{kind}_decoder_{epoch}_generalize_{split_by}{hit_label}{state_label}_({align} aligned).pkl'

            # Under construction (use at your own risk)
            elif kind == 'epoch_ortho':
                pred, pred_err, acc, acc_null = \
                    epoch_cross_decoder_ORTHO(bins, epoch, epoch_ortho, all_psth, df_behavior[col], n_shuffles)
                filename = f'epoch_cross_decoder_ORTHO_{epoch}_{state_label}.pkl'
            else:
                raise ValueError("Kind must be 'within', 'cross', 'epoch', or 'epoch_split'")

            results['pred'].append(pred)
            results['pred_err'].append(pred_err)
            results['acc'].append(acc)
            results['acc_null'].append(acc_null)
            results['bins'] = bins

        except Exception as e:
            print(f'An error occurred in session {ephys_ids[i]}: {e}')
            error_sessions.append(ephys_ids[i])
            traceback.print_exc()
            continue

    # Plot decoding accuracy for each session
    if plot:
        if kind == 'within':
            pass
        elif kind == 'cross':
            pass
        elif kind == 'epoch':
            pass
        elif kind == 'epoch_split':
            pass

    if save:
        os.chdir(folder_parent)
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

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
    # plt.figure(constrained_layout=True)
    plt.axvline(0, color='tab:red', linestyle='-')  # Stimulus onset
    plt.axvline(1, color='tab:gray', linestyle='-')  # Go cue

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute p-values and z-scores relative to the null distribution
    if z_null:
        z_scores = null_zscore(acc, acc_null)
        p_values = p_val(acc, acc_null)
        significant_region = p_values < 0.05  # When assessing significance of single sessions use p < 0.05
        plt.plot(bin_centers, z_scores, label='Z acc.')
        plt.fill_between(bin_centers, z_scores, where=significant_region, edgecolor='none',
                         alpha=0.25, label='α < .05')
        plt.axhline(0, color='tab:gray', linestyle='--')  # Chance level
        ylabel = 'Z-score'
    else:
        acc_mean = acc.mean(axis=0)
        acc_sem = sem(acc, axis=0)
        acc_null_mean = acc_null.mean(axis=0)
        acc_null_band = np.percentile(acc_null, [2.5, 97.5], axis=0)  # The 95% confidence interval of the shuffles
        # plt.plot(-np.mean(abs(pred_err), axis=0)+1)  # Equivalent
        plt.plot(bin_centers, acc_mean, label='Acc.')
        plt.fill_between(bin_centers, acc_mean - acc_sem, acc_mean + acc_sem, edgecolor='none', alpha=0.25,
                         label='Acc. s.e.m.')
        plt.plot(bin_centers, acc_null_mean, color='tab:gray', linestyle='--', label='Null mean')  # Chance level (0.5)
        plt.fill_between(bin_centers, acc_null_band[0], acc_null_band[1], color='tab:gray', edgecolor='none', alpha=0.25,
                         label='Null 95% CI')
        ylabel = 'Accuracy'

    plt.xlim(bins[0], bins[-1])
    plt.ylim(None, 1)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    # plt.title(f'Decoding accuracy\n'
    #           f'{df_behavior.Subject.unique()[0]}, {acc.shape[0]} trials')
    # plt.legend(frameon=False)
    sns.despine()


def plot_mean_within_decoder(results, errorbar='ci', z_null=False):
    """
    Plot the mean decoding accuracy across all sessions.
    :param results: dict with decoding results for each session
    :param errorbar: type of error bar (ci=Confidence Interval, sem=Standard Error of the Mean). Only if z_null=False
    :param z_null: whether to Z-score the decoding accuracy by the null distribution of accuracy
    """

    # plt.figure(constrained_layout=True)
    plt.axvline(0, color='tab:gray', linestyle='-')  # Stimulus onset
    plt.axvline(0.5, color='tab:gray', linestyle='--')  # Delay
    plt.axvline(1, color='tab:gray', linestyle='-')  # Go cue

    # Compute the mean accuracy across all sessions
    acc_mean = [results['acc'][i].mean(axis=0) for i in range(len(results['acc']))]
    acc_mean = np.array(acc_mean)
    acc_null_mean = [results['acc_null'][i].mean(axis=0) for i in range(len(results['acc_null']))]
    acc_null_mean = np.array(acc_null_mean)
    n_trials = np.sum([results['acc'][i].shape[0] for i in range(len(results['acc']))])
    bins = results['bins']
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if z_null:
        z_scores = [null_zscore(results['acc'][i], results['acc_null'][i]) for i in range(len(results['acc']))]
        z_scores_mean = np.mean((z_scores), axis=0)
        plt.plot(bin_centers, z_scores_mean, color='tab:blue', label='Z acc.')
        p_values = p_val(acc_mean, acc_null_mean)
        significant_region = p_values < 0.05  # When assessing significance across sessions use p < 0.05
        # significant_region = np.abs(z_scores_mean) >= 1.96  # When assessing significance across sessions use 1.96
        plt.fill_between(bin_centers, z_scores_mean, where=significant_region, edgecolor='none', color='tab:blue',
                         alpha=0.25, label='α < .05')
        plt.axhline(0, color='tab:gray', linestyle='--')  # Chance level
        ylabel = 'Z-score'

    else:
        acc_sem = sem(acc_mean, axis=0)
        acc_mean = np.mean(acc_mean, axis=0)
        acc_null_sem = sem(acc_null_mean, axis=0)
        acc_null_mean = np.mean(acc_null_mean, axis=0)
        acc_CI = 1.96 * acc_sem  # 95% confidence interval
        acc_null_CI = 1.96 * acc_null_sem  # 95% confidence interval

        if errorbar == 'ci':
            acc_band = (acc_mean - acc_CI, acc_mean + acc_CI)
            acc_null_band = (acc_null_mean - acc_null_CI, acc_null_mean + acc_null_CI)
            acc_band_label = 'Acc. 95% CI'
            acc_null_band_label = 'Acc. null 95% CI'
        elif errorbar == 'sem':
            acc_band = (acc_mean - acc_sem, acc_mean + acc_sem)
            acc_null_band = (acc_null_mean - acc_null_sem, acc_null_mean + acc_null_sem)
            acc_band_label = 'Acc. SEM'
            acc_null_band_label = 'Acc. null SEM'

        # Plot the mean decoding accuracy across all sessions
        plt.plot(bin_centers, acc_mean, color='tab:blue', label='Acc.')
        plt.fill_between(bin_centers, acc_band[0], acc_band[1], color='tab:blue', edgecolor='none',
                         alpha=0.25)#, label=acc_band_label)

        # Plot the mean null accuracy across all sessions (chance level)
        plt.plot(bin_centers, acc_null_mean, ls='--', color='tab:gray', label='Acc. null')
        plt.fill_between(bin_centers, acc_null_band[0], acc_null_band[1], color='tab:gray',
                         edgecolor='none', alpha=0.25)#, label=acc_null_band_label)

        # Plot the individual sessions accuracy
        for _ in range(len(results['acc'])):
            plt.plot(bin_centers, np.mean(results['acc'][_], axis=0), color='tab:blue', alpha=0.1)

        # # Plot the individual sessions null accuracy (chance level)
        # for _ in range(len(results['acc_null'])):
        #     plt.plot(bin_centers, np.mean(results['acc_null'][_], axis=0), ls='--', color='tab:gray')

        ylabel = 'Accuracy'

    plt.xlim(bins[0], bins[-1])
    plt.ylim(None, 1)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    # plt.title(f"Decoding accuracy\n"
    #           f"{subject}, {len(results['acc'])} sessions, {n_trials} trials")
    sns.despine()
    # plt.legend(frameon=False)


def plot_cross_decoder(bins, acc, acc_null, z_null=True):
    """
    Plot the prediction error for the cross-temporal decoder.
    :param bins: 1D array with time bins
    :param acc: 3D array with decoding accuracy (trials x time x time)
    :param acc_null: 3D array with null distribution of accuracy (shuffles x time x time)
    :param z_null: whether to Z-score the decoding accuracy by the null distribution of accuracy
    """

    bin_width = np.diff(bins).mean()
    extent = [bins[0] - bin_width / 2, bins[-1] + bin_width / 2,
              bins[0] - bin_width / 2, bins[-1] + bin_width / 2]

    # plt.figure(constrained_layout=True)

    if z_null:
        z_score = null_zscore(acc, acc_null)
        plt.imshow(z_score, origin='lower', cmap='RdBu_r', norm=CenteredNorm(), extent=extent)  # abs needed?
        axislabel = ' - Z-score'
    else:
        plt.imshow(np.mean(acc, axis=0), origin='lower', extent=extent)  # abs needed?
        axislabel = ''

    plt.colorbar()

    plt.axhline(0, color='tab:gray', linestyle='-')  # Stimulus
    plt.axvline(0, color='tab:gray', linestyle='-')  # Stimulus
    plt.axhline(0.5, color='tab:gray', linestyle='--')  # Delay
    plt.axvline(0.5, color='tab:gray', linestyle='--')  # Delay
    plt.axhline(1, color='tab:gray', linestyle='-')  # Go cue
    plt.axvline(1, color='tab:gray', linestyle='-')  # Go cue

    plt.xticks(bins[::10], np.round(bins[::10], 1).astype(int))
    plt.yticks(bins[::10], np.round(bins[::10], 1).astype(int))
    plt.xlabel('Test time (s)' + axislabel)
    plt.ylabel('Train time (s)' + axislabel)
    # plt.title('Cross temporal decoder')
    # plt.title(f'Decoding accuracy\n'
    #           f'{df_behavior.Subject.unique()[0]}, {acc.shape[0]} trials')
    sns.despine()


def plot_mean_cross_decoder(results, align='stim', z_null=True):
    """
    Plot the mean decoding accuracy across all sessions.
    :param results: dict with decoding results for each session
    :return:
    """

    # plt.figure(constrained_layout=True)

    # Compute the mean accuracy across all sessions
    acc_mean = [results['acc'][i].mean(axis=0) for i in range(len(results['acc']))]
    acc_mean = np.array(acc_mean)
    acc_mean = np.mean(acc_mean, axis=0)
    # acc_null_mean = [results['acc_null'][i].mean(axis=0) for i in range(len(results['acc_null']))]
    # acc_null_mean = np.array(acc_null_mean)
    # acc_null_mean = np.mean(acc_null_mean, axis=0)
    n_trials = np.sum([results['acc'][i].shape[0] for i in range(len(results['acc']))])
    bins = results['bins']
    bin_width = np.diff(bins).mean()
    extent = [bins[0] - bin_width / 2, bins[-1] + bin_width / 2,
              bins[0] - bin_width / 2, bins[-1] + bin_width / 2]
    x_min, x_max = extent[0], extent[1]

    if z_null:
        z_scores = [null_zscore(results['acc'][i], results['acc_null'][i]) for i in range(len(results['acc']))]
        z_scores_mean = np.mean((z_scores), axis=0)
        plt.imshow(z_scores_mean, origin='lower', cmap='RdBu_r', norm=CenteredNorm(), extent=extent)
    else:
        plt.imshow(acc_mean, origin='lower', extent=extent)  # abs needed?

    plt.colorbar(label='Z-score', )

    color = 'tab:gray'
    plt.axhline(0, color=color, linestyle='-')  # Stimulus / First lick
    plt.axvline(0, color=color, linestyle='-')  # Stimulus / First lick
    plt.axhline(1, color=color, linestyle='-')  # Go cue / ITI
    plt.axvline(1, color=color, linestyle='-')  # Go cue / ITI
    if align =='stim':
        plt.axhline(0.5, color=color, linestyle='--')  # Delay
        plt.axvline(0.5, color=color, linestyle='--')  # Delay

    # Add labeled rectangles for epoch slices
    if align == 'stim':
        epochs = {
            'stim': {'range': (0, 0.1), 'color': 'tab:blue', 'label': 'S'},
            'delay': {'range': (0.9, 1), 'color': 'tab:orange', 'label': 'D'},
            'resp': {'range': (1.85, 1.95), 'color': 'tab:green', 'label': 'R'}
        }
    # elif align == 'resp':
    #     epochs = {
    #         'stim': {'range': (-1, -0.9), 'color': 'tab:blue', 'label': 'Stimulus'},
    #         'delay': {'range': (-0.1, 0), 'color': 'tab:orange', 'label': 'Delay'},
    #         'resp': {'range': (0, 0.1), 'color': 'tab:green', 'label': 'Response'}
    #     }

        ax = plt.gca()
        for name, props in epochs.items():
            start, end = props['range']
            color = props['color']
            label = props['label']

            rect = patches.Rectangle(
                xy=(x_min, start),
                width=x_max - x_min,
                height=end - start,
                edgecolor=color,
                facecolor='none',
                zorder=2
            )
            ax.add_patch(rect)

            ax.text(
                x=x_min,
                y=start + 0.15,
                s=label,
                color=color,
                ha='left',
            )

    first_tick = np.ceil(bins[0])  # Round up to the nearest integer
    last_tick = np.floor(bins[-1])  # Round down to the nearest integer
    ticks = np.arange(first_tick, last_tick + 1, 1)  # Create ticks at every integer value
    plt.xticks(ticks, ticks.astype(int))
    plt.yticks(ticks, ticks.astype(int))

    # plt.xticks(bins[::10], np.round(bins[::10], 1).astype(int))
    # plt.yticks(bins[::10], np.round(bins[::10], 1).astype(int))
    plt.xlabel('Test time (s)')
    plt.ylabel('Train time (s)')
    # plt.title(f"Decoding accuracy\n"
    #           f" {subject}, {len(results['acc'])} sessions, {n_trials} trials")
    sns.despine()


def plot_epoch_cross_decoder(bins, acc, acc_null, epoch=None, z_null=True):

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
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, acc_mean, label=label)
    plt.fill_between(bin_centers, acc_mean - acc_sem, acc_mean + acc_sem, edgecolor='none', alpha=0.25)
    plt.plot(bin_centers, acc_null_mean, color='tab:gray', linestyle='--', label='Null')
    plt.fill_between(bin_centers, acc_null_mean - acc_null_sem, acc_null_mean + acc_null_sem, edgecolor='none',
                     color='tab:gray', alpha=0.25)
    plt.legend(frameon=False)
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.title(f'Decoding accuracy\n'
              f' {n_trials} trials')
    sns.despine()


def plot_mean_epoch_cross_decoder(results, epoch=None, engagement=None, errorbar='ci', z_null=True):
    """
    Plot the mean epoch cross temporal decoding accuracy across all sessions.
    :param results: dict with decoding results for each session
    :param z_null:
    """

    if epoch == 'stim':
        color = 'tab:blue'
        label = 'Stimulus'
    elif epoch == 'delay':
        color = 'tab:orange'
        label = 'Delay'
    elif epoch == 'resp':
        color = 'tab:green'
        label = 'Response'
    elif epoch == 'first_lick':
        color = 'darkgreen'
        label = 'First lick'
    elif epoch == 'mid_lick':
        color = 'lightgreen'
        label = 'Mid lick'

    if engagement is not None:
        if engagement == 0:
            color = 'tab:gray'
            label = 'Disengaged'
        elif engagement == 1:
            label = 'Engaged'

    acc_mean = [np.nanmean(results['acc'][i], axis=0) for i in range(len(results['acc']))]
    acc_mean = np.array(acc_mean)
    acc_sem = sem(acc_mean, axis=0, nan_policy='omit')
    acc_mean = np.nanmean(acc_mean, axis=0)
    acc_CI = 1.96 * acc_sem  # 95% confidence interval

    acc_null_mean = [np.nanmean(results['acc_null'][i], axis=(0, 2)) for i in range(len(results['acc_null']))]
    acc_null_mean = np.array(acc_null_mean)
    acc_null_sem = sem(acc_null_mean, axis=0, nan_policy='omit')
    acc_null_mean = np.nanmean(acc_null_mean, axis=0)
    acc_null_CI = 1.96 * acc_null_sem  # 95% confidence interval

    if errorbar == 'ci':
        acc_band = (acc_mean - acc_CI, acc_mean + acc_CI)
        acc_null_band = (acc_null_mean - acc_null_CI, acc_null_mean + acc_null_CI)
        acc_band_label = 'Acc. 95% CI'
        acc_null_band_label = 'Acc. null 95% CI'
    elif errorbar == 'sem':
        acc_band = (acc_mean - acc_sem, acc_mean + acc_sem)
        acc_null_band = (acc_null_mean - acc_null_sem, acc_null_mean + acc_null_sem)
        acc_band_label = 'Acc. SEM'
        acc_null_band_label = 'Acc. null SEM'

    bins = results['bins']
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.axvline(0, color='tab:gray', linestyle='-')  # Stimulus onset
    plt.axvline(0.5, color='tab:gray', linestyle='--')  # Delay
    plt.axvline(1, color='tab:gray', linestyle='-')  # Go cue

    # n_trials = np.sum([results['pred'][i].shape[0] for i in range(len(results['acc']))])
    # plt.figure(constrained_layout=True)
    plt.plot(bin_centers, acc_mean, color=color, label=label)
    plt.fill_between(bin_centers, acc_band[0], acc_band[1], color=color, edgecolor='none', alpha=0.25)

    plt.plot(bin_centers, acc_null_mean, linestyle='--', color=color)
    plt.fill_between(bin_centers, acc_null_band[0], acc_null_band[1], color=color, edgecolor='none',
                        alpha=0.25)

    plt.legend(frameon=False)
    plt.xlim(bins[0], bins[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.ylim(None, 1)
    # plt.title(f'Decoding accuracy\n'
    #           f'{subject}, {len(results["acc"])} sessions, {n_trials} trials')
    sns.despine()


def plot_mean_epoch_cross_decoder_split(results, epoch=None, split='hit', errorbar='ci', z_null=True):
    """
    Plot the mean epoch cross temporal decoding accuracy across all sessions.
    :param results: dict with decoding results for each session
    :param z_null: whether to Z-score the decoding accuracy by the null distribution of accuracy
    """

    # Plot mean accuracy epoch stimulus decoder split by split_by

    # Condition 0
    acc0_mean = np.nanmean([np.nanmean(results['acc'][i][0], axis=0) for i in range(len(results['acc']))], axis=0)
    acc0_sem = sem([np.nanmean(results['acc'][i][0], axis=0) for i in range(len(results['acc']))], axis=0)
    acc_null0_mean = np.nanmean(
        [(np.nanmean(results['acc_null'][i][0], axis=(0, 2))) for i in range(len(results['acc_null']))], axis=0)
    acc_null0_sem = sem([np.nanmean(results['acc_null'][i][0], axis=(0, 2)) for i in range(len(results['acc_null']))],
                        axis=0)
    acc0_CI = 1.96 * acc0_sem  # 95% confidence interval
    acc_null0_CI = 1.96 * acc_null0_sem  # 95% confidence interval

    # Condition 1
    acc1_mean = np.nanmean([np.nanmean(results['acc'][i][1], axis=0) for i in range(len(results['acc']))], axis=0)
    acc1_sem = sem([np.nanmean(results['acc'][i][1], axis=0) for i in range(len(results['acc']))], axis=0)
    acc_null1_mean = np.nanmean(
        [(np.nanmean(results['acc_null'][i][1], axis=(0, 2))) for i in range(len(results['acc_null']))], axis=0)
    acc_null1_sem = sem([np.nanmean(results['acc_null'][i][0], axis=(0, 2)) for i in range(len(results['acc_null']))],
                        axis=0)
    acc1_CI = 1.96 * acc1_sem  # 95% confidence interval
    acc_null1_CI = 1.96 * acc_null1_sem  # 95% confidence interval

    if errorbar == 'ci':
        acc0_band = (acc0_mean - acc0_CI, acc0_mean + acc0_CI)
        acc1_band = (acc1_mean - acc1_CI, acc1_mean + acc1_CI)
        acc_null0_band = (acc_null0_mean - acc_null0_CI, acc_null0_mean + acc_null0_CI)
        acc0_band_label = 'Acc0. 95% CI'
        acc_null0_band_label = 'Acc. null 95% CI'
    elif errorbar == 'sem':
        acc0_band = (acc0_mean - acc0_sem, acc0_mean + acc0_sem)
        acc1_band = (acc1_mean - acc1_sem, acc1_mean + acc1_sem)
        acc_null1_band = (acc_null1_mean - acc_null1_sem, acc_null1_mean + acc_null1_sem)
        acc1_band_label = 'Acc1. SEM'
        acc_null1_band_label = 'Acc1. null SEM'

    # Flip it to make it easier to compare
    chance = 0.5
    acc0_mean = chance + (chance - acc0_mean)
    acc0_band = (chance + (chance - acc0_band[0]), chance + (chance - acc0_band[1]))
    acc_null0_mean = chance + (chance - acc_null0_mean)
    acc_null0_band = (chance + (chance - acc_null0_band[0]), chance + (chance - acc_null0_band[1]))

    if split == 'hit':
        colors = ('tab:red', 'tab:green')
        labels = ('Error', 'Correct')
    elif split == 'engagement':
        if epoch == 'stim':
            colors = ('tab:gray', 'tab:blue')
        elif epoch == 'delay':
            colors = ('tab:gray', 'tab:orange')
        elif epoch == 'resp':
            colors = ('tab:gray', 'tab:green')
        labels = ('Disengaged', 'Engaged')

    # plt.figure(constrained_layout=True)

    bins = results['bins']
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.axvline(0, color='tab:gray', linestyle='-')  # Stimulus onset
    plt.axvline(0.5, color='tab:gray', linestyle='--')  # Delay
    plt.axvline(1, color='tab:gray', linestyle='-')  # Go cue

    # Condition 0 (error/disengaged)
    plt.plot(bin_centers, acc0_mean, color=colors[0], label=labels[0])
    plt.fill_between(bin_centers, acc0_band[0], acc0_band[1], color=colors[0],
                     edgecolor='none', alpha=0.25)
    plt.plot(bin_centers, acc_null0_mean, color=colors[0], linestyle='--')
    # plt.fill_between(bin_centers, acc_null0_band[0], acc_null0_band[1], color=colors[0], edgecolor='none',
    #                  alpha=0.25)

    # Condition 1 (correct/engaged). Negative to flip it and make it easier to compare
    plt.plot(bin_centers, acc1_mean, color=colors[1], label=labels[1])
    plt.fill_between(bin_centers, acc1_band[0], acc1_band[1], color=colors[1],
                     edgecolor='none', alpha=0.25)
    plt.plot(bin_centers, acc_null1_mean, color=colors[1], linestyle='--')
    # plt.fill_between(bin_centers, acc_null1_band[0], acc_null1_band[1], color='tab:green', edgecolor='none',
    #                     alpha=0.25)

    plt.xlim(bins[0], bins[-1])
    # plt.ylim(None, 1)
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.legend(frameon=False)
    sns.despine()


########################################################################################################################
# UNDER CONSTRUCTION
# Please wear safety equipment. Any resemblance to real cool effects might be purely acidental. I do not take
# responsibility for any damage caused by the use of this code. Use at your own risk.
########################################################################################################################

# Testing split according to engagement
# i = 0
# path_behavior = get_behavior_id(ephys_ids[i])
# df_behavior = parse_v2(path_behavior)
# disengagement = find_disengaged(df_behavior, plot=False)  # Find trial when disengagement happens
# engaged = (df_behavior.Trial <= disengagement).astype(int)  # Compare trials to disengagement (no need for i)
# df_behavior['Engaged'] = engaged
# bins = np.load(folder_parent / ephys_ids[i] / 'bins.npy')
# all_psth = np.load(folder_parent / ephys_ids[i] / 'all_psth.npy')
# # Correct trials only
# correct_idx = df_behavior[df_behavior.Hit == 1].index
# df_behavior = df_behavior.iloc[correct_idx].reset_index(drop=True)
# all_psth = all_psth[correct_idx]
#
# epoch_cross_decoder_split_TEST(bins, df_behavior.Engaged, epoch='stim', X=all_psth, y=df_behavior.Side, n_shuffles=100)


@timer
def epoch_cross_decoder_ORTHO(bins, epoch=None, epoch_ortho=None, X=np.zeros((1, 1, 1)), y=np.zeros((1, 1)),
                              n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary stimulus condition from neural data using K-fold
    cross-validation across trials and across time bins.
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary stimulus condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    def sigmoid(z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    n_splits = 5

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape
    pred = np.empty((n_trials, n_bins)) * np.nan
    pred_err = np.empty((n_trials, n_bins)) * np.nan
    # acc = np.empty((n_splits, n_bins))  # Store per fold and bin
    acc = np.empty((n_trials, n_bins)) * np.nan  # Store per trial and bin
    acc_ortho = np.empty((n_trials, n_bins)) * np.nan  # Store per trial and bin
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

    # Define orthogonalization epoch
    if epoch_ortho == 'stim':
        epoch_start_idx_ortho = np.where(np.round(bins, 1) == 0)[0][0]  # Find index where stim_onset (0) is
        epoch_end_idx_ortho = np.where(np.round(bins, 1) == 0.2)[0][0]  # Find index where delay (0.5) is
    elif epoch_ortho == 'delay':
        epoch_start_idx_ortho = np.where(np.round(bins, 1) == 0.8)[0][0]  # Find index where delay (0.5) is
        epoch_end_idx_ortho = np.where(np.round(bins, 1) == 1)[0][0]  # Find index where go cue is in bins
    elif epoch_ortho == 'resp':
        epoch_start_idx_ortho = np.where(np.round(bins, 1) == 1.8)[0][0]  # Find index where go cue is in bins
        epoch_end_idx_ortho = np.where(np.round(bins, 1) == 2)[0][0]  # Find index where go cue is in bins

    # Split trials into training and testing sets (each fold gets a unique test set to prevent overfitting)
    # for train_index, test_index in kf.split(X):
    for k, (train_index, test_index) in enumerate(skf.split(X, y)):

        # Cross thinghy happens here
        X_train = np.mean(X[train_index, epoch_start_idx:epoch_end_idx], axis=1)
        X_train_ortho = np.mean(X[train_index, epoch_start_idx_ortho:epoch_end_idx_ortho], axis=1)
        y_train = y[train_index]
        # Apply z-scoring normalization to the current time bin_train's data (otherwise might not converge)
        X_train = scaler.fit_transform(X_train)  # Fit and transform on the training set
        X_train_ortho = scaler.fit_transform(X_train_ortho)  # Fit and transform on the training set

        # Train decoder (logistic regression) on the current time bin_train’s neural activity
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        clf_ortho = LogisticRegression()
        clf_ortho.fit(X_train_ortho, y_train)

        # ortogonaliye weights
        weights = np.squeeze(clf.coef_)
        weights_ortho = np.squeeze(clf_ortho.coef_)
        # v2 = w1 - np.dot(v1, w2)/(v1, v1) * v1
        weights_orthogonalized = weights - np.dot(weights_ortho, weights) / np.dot(weights_ortho, weights_ortho) * weights_ortho
        # How to Compute the Intercept for the Orthogonalized Model?
        intercept_ortho = clf.intercept_[0] - (clf_ortho.intercept_[0] * clf.intercept_[0] / (clf_ortho.intercept_[0] ** 2)) * clf_ortho.intercept_[0]

        # Loop over each time bin_train
        for bin_test in range(n_bins):

            # Define train and testing set for the current time bin_train and fold
            X_test = X[test_index, bin_test]
            y_test = y[test_index]

            # Apply z-scoring normalization to the current time bin_train's data (otherwise might not converge)
            X_test = scaler.transform(X_test)  # Only transform the test set using the same scaler

            # Evaluate decoder
            y_pred = clf.predict(X_test)  # Predicts the stimulus category for test trials

            # predict ortho
            z = np.dot(X_test, weights_orthogonalized) + intercept_ortho  # Linear combination
            probabilities = sigmoid(z)  # Apply sigmoid
            y_pred_ortho = (probabilities >= 0.5).astype(int)

            # Predicts the stimulus category for test trials
            # y_acc = accuracy_score(y_test, y_pred)  # Computes accuracy for each fold
            y_acc = (y_pred == y_test).astype(int)  # Computes accuracy per trial
            y_acc_ortho = (y_pred_ortho == y_test).astype(int)  # Computes accuracy per trial
            # print(f"Accuracy: {y_acc:.2f}")

            # Store results
            pred[test_index, bin_test] = y_pred  # Predicted stimulus condition for each test trial at each time
            # bin_train
            pred_err[test_index, bin_test] = y_pred - y_test  # Difference between predicted and actual labels
            # acc[k, bin_test] = y_acc  # Accuracy for each test trial at each time bin
            acc[test_index, bin_test] = y_acc  # Accuracy for each test trial at each time bin
            acc_ortho[test_index, bin_test] = y_acc_ortho  # Accuracy for each test trial at each time bin

            # Compute null distribution by shuffling the labels and evaluating accuracy
            y_test_shuffled = y_test.values.copy()
            for _ in range(n_shuffles):
                np.random.shuffle(y_test_shuffled)
                # acc_null[_, bin_test] = accuracy_score(y_test_shuffled, y_pred)
                acc_null[test_index, bin_test, _] = (y_pred == y_test_shuffled).astype(int)  # Computes accuracy per trial
                # pred_err_null[test_index, bin_test, _] = y_pred - y_test_shuffled  # Difference between predicted and
                # actual labels

            acc = acc_ortho

    return pred, pred_err, acc, acc_null

# i = 0
# path_behavior = get_behavior_id(ephys_ids[i])
# df_behavior = parse_v2(path_behavior)
# disengagement = find_disengaged(df_behavior, plot=False)  # Find trial when disengagement happens
# engaged = (df_behavior.Trial <= disengagement).astype(int)  # Compare trials to disengagement (no need for i)
# df_behavior['Engaged'] = engaged
# bins = np.load(folder_parent / ephys_ids[i] / 'bins.npy')
# all_psth = np.load(folder_parent / ephys_ids[i] / 'all_psth.npy')
# # Correct trials only
# correct_idx = df_behavior[df_behavior.Hit == 1].index
# df_behavior = df_behavior.iloc[correct_idx].reset_index(drop=True)
# all_psth = all_psth[correct_idx]
# epoch_cross_decoder_ORTHO(bins, epoch='stim', epoch_ortho='delay', X=all_psth, y=df_behavior.Side,
#                           n_shuffles=100)


# stim_len = df.StimLen.unique()[0]
# delay = df.Delay.unique()[0]
# resp_win = df.RespWinLen.unique()[0]
# timeout = df.Timeout.unique()[0]
# iti = df.ITI.unique()[0]
# load_time = 2.5
#
# total = stim_len + delay + resp_win + timeout + iti + load_time
# print(f'Trial total time: {total}s')


subjects = ['000', '007', '009']  # Removed 001 (all sessions bad)
folder_parent = Path.home() / 'data'

for subj in subjects:
    # preprocess_subject(subj)

    # X decoder
    # mean_decoder(subj, what='stim', align='resp', kind='cross', epoch=None, epoch_ortho=None, split_by=None, drop_miss=True,
    #              hit_only=True, engagement=1, n_shuffles=100, plot=False, save=True)

    # Epoch decoders (align to first lick)
    # mean_decoder(subj, what='stim', align='resp', kind='epoch', epoch='first_lick', epoch_ortho=None, split_by=None,
    #              drop_miss=True, hit_only=True, engagement=1, n_shuffles=100, plot=False, save=True)

    # Split by outcome
    # mean_decoder(subj, what='stim', kind='epoch_generalize', epoch='stim', epoch_ortho=None, split_by='Hit', drop_miss=True,
    #                  hit_only=False, engagement=1, n_shuffles=100, plot=False, save=True)
    # mean_decoder(subj, what='stim', kind='epoch_generalize', epoch='delay', epoch_ortho=None, split_by='Hit', drop_miss=True,
    #                  hit_only=False, engagement=1, n_shuffles=100, plot=False, save=True)
    # mean_decoder(subj, what='stim', kind='epoch_generalize', epoch='resp', epoch_ortho=None, split_by='Hit', drop_miss=True,
    #                  hit_only=False, engagement=1, n_shuffles=100, plot=False, save=True)

    # Split by engagement
    mean_decoder(subj, what='stim', kind='epoch_split', epoch='stim', epoch_ortho=None, split_by='Engaged', drop_miss=True,
                     hit_only=True, engagement=None, n_shuffles=100, plot=False, save=True)
    mean_decoder(subj, what='stim', kind='epoch_split', epoch='delay', epoch_ortho=None, split_by='Engaged', drop_miss=True,
                     hit_only=True, engagement=None, n_shuffles=100, plot=False, save=True)
    mean_decoder(subj, what='stim', kind='epoch_split', epoch='resp', epoch_ortho=None, split_by='Engaged', drop_miss=True,
                     hit_only=True, engagement=None, n_shuffles=100, plot=False, save=True)


# Paralelization test
# subjects = ['000', '007', '009']
# folder_parent = Path.home() / 'data'
# def run_fun(subj):
#     # mean_decoder(subj, what='stim', align='resp', kind='cross', epoch=None, epoch_ortho=None,
#     #              split_by=None, drop_miss=True, hit_only=True, engagement=1, n_shuffles=100,
#     #              plot=False, save=True)
#     preprocess_subject(subj, align='stim', time_win=[-1, 3], bin_size=0.1)
# Parallel(n_jobs=-1)(delayed(run_fun)(subj) for subj in subjects)