import os
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import threadpoolctl
threadpoolctl.threadpool_info()

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import matplotlib.patches as patches
import seaborn as sns
from scipy.stats import sem, zscore
import pickle
import traceback
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm

from my_fun.my_fun import timer, filter_behavior
from ephys.preprocessing import *
from ephys.analysis import *


# Neuromatch tutorial: https://compneuro.neuromatch.io/tutorials/W1D5_DeepLearning/student/W1D5_Tutorial1.html

########################################################################################################################

def preprocess_subject(subject, align='stim', overwrite=False):
    """
    Preprocess all ephys and behavioral sessions for a given subject and save the results in a folder.
    :param subject: subject ID (str)
    :return: None
    """

    if align == 'stim':
        time_win = [-1, 4]
        bin_size = 0.1
    elif align == 'resp':
        time_win = [-1, 2]
        bin_size = 0.05

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

        npy_exists = any(f.endswith(f'{align}.npy') for f in os.listdir(folder_child))
        csv_exists = any(f.endswith('.csv') for f in os.listdir(folder_child))

        if npy_exists and csv_exists and not overwrite:
            print('Files exist in folder and overwrite=False. Skipping...')
        else:
            try:
                print('Files do not exist in folder or overwrite=True. Proceeding...')
                preprocessed = preprocess(ephys_ids[i])
                df_ttl, df_behavior, n_trials, df_spikes, cluster_info, timeline = preprocessed
                bins, all_psth = get_all_psth(cluster_info, df_spikes, df_ttl, n_trials, align=align, time_win=time_win,
                                              bin_size=bin_size)
                np.save(folder_child / f'bins_{align}.npy', bins)
                np.save(folder_child / f'all_psth_{align}.npy', all_psth)
                filename = df_behavior.Session.unique()[0] + '.csv'
                df_behavior.to_csv(folder_child / filename, index=False)
            except Exception as e:
                print(f'An error occurred: {e}')
                # traceback.print_exc()
                error_sessions.append(ephys_ids[i])

    print(f'Sessions not preprocessed: {error_sessions}')

    return error_sessions


def summary_behavior(df):
    """
    Summarize behavioral performance by session.
    :param df: DataFrame with behavioral data
    :return: summary DataFrame
    """

    df = df[df.Miss == 0]  # Exclude missed trials

    summary = (df
        .groupby('Session')
        .apply(lambda x: pd.Series({
            'TotalLeft': (x['Side'] == 0).sum(),
            'TotalRight': (x['Side'] == 1).sum(),
            'Total': len(x),
            'ErrorLeft': ((x['Side'] == 0) & (x['Hit'] == 0)).sum(),
            'ErrorRight': ((x['Side'] == 1) & (x['Hit'] == 0)).sum(),
            'TotalError': (x['Hit'] == 0).sum(),
            'CorrectLeft': ((x['Side'] == 0) & (x['Hit'] == 1)).sum(),
            'CorrectRight': ((x['Side'] == 1) & (x['Hit'] == 1)).sum(),
            'TotalCorrect': (x['Hit'] == 1).sum(),

        }))
        .reset_index()
    )

    summary['AccLeft'] = summary['CorrectLeft'] / summary['TotalLeft']
    summary['AccRight'] = summary['CorrectRight'] / summary['TotalRight']
    summary['TotalAcc'] = summary['TotalCorrect'] / summary['Total']

    # pd.set_option('display.max_columns', None)  # Show all columns
    # pd.set_option('display.width', 0)  # No line wrapping
    # print(summary)

    return summary


def get_all_beh(subject, glmhmm=False):
    """
    Get and concatenate behavioral data from all ephys sessions for a given subject.
    :param subject: subject ID (str)
    """

    folder_parent = Path.home() / 'data' / subject
    ephys_ids = get_ephys_sessions(subject)
    df = pd.DataFrame()

    if glmhmm:
        # Load GLM-HMM fit
        path_glmhmm = (Path.home() / 'PycharmProjects' / 'glmhmm' / 'TEST' / '2s_3cov' / 'Ephys' / subject).with_suffix('.csv')

    for i in range(len(ephys_ids)):

        # Create child folder within parent folder for each ephys_id with its name if it doesn't exist
        folder_child = folder_parent / ephys_ids[i]
        folder_child.mkdir(parents=True, exist_ok=True)
        os.chdir(folder_child)

        # .npy files (spike counts)
        if any(f.endswith('.npy') for f in os.listdir(folder_child)):
            print("'.npy' files exist in folder. Proceeding...")
            print(f'Processing session {i + 1}/{len(ephys_ids)}: {ephys_ids[i]}')
            folder_child = folder_parent / ephys_ids[i]
            filename_behavior = [f for f in os.listdir(folder_child) if f.endswith('.csv')][0]  # Assume only one .csv file
            path_behavior = folder_child / filename_behavior
            df_behavior = pd.read_csv(path_behavior)

            if glmhmm:
                df_glmhmm = pd.read_csv(path_glmhmm)
                session = df_behavior.Session.unique()[0]
                df_glmhmm = df_glmhmm[df_glmhmm.Session == session].reset_index(drop=True)
                df_behavior = df_behavior[df_behavior.Miss == 0].reset_index(drop=True)
                print(len(df_behavior))
                print(len(df_glmhmm))
                # assert len(df_glmhmm) == len(df_behavior)  # Check sessions match
                df_behavior = df_glmhmm

            df = pd.concat([df_behavior, df], ignore_index=True)

        else:
            print('There are no spike count files in the folder. Skipping...')

    return df


def get_rec_side(ephys_id):
    df = pd.read_csv(r'/home/alexis/PycharmProjects/ephys/RECside.csv')
    df.Subject = df.Subject.astype(int).astype(str).str.zfill(3)  # 00X pad subjects ID
    # df = df[df.Subject.isin(subjects)]
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True).dt.date
    subject = ephys_id[:3]
    date = pd.to_datetime(ephys_id.split('_')[1]).date()
    df['RECside'] = df['RECside'].map({'L': 0, 'R': 1})
    rec_side = df.loc[(df.Subject == subject)&(df.Date == date), 'RECside'].iloc[0]
    return rec_side


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
def within_decoder(X=np.empty((1, 1, 1)), y=np.empty((1, 1)), n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary condition from neural data using K-fold cross-validation.
    Train and test within the same time within_bin.
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    # Cross-validate results
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Stratified cross-validation
    n_splits = skf.get_n_splits()

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape
    pred = np.full((n_trials, n_bins), np.nan)
    pred_err = np.full((n_trials, n_bins), np.nan)
    acc = np.full((n_trials, n_bins), np.nan)
    acc_null = np.full((n_shuffles, n_bins), np.nan)
    # acc_null = np.zeros((n_shuffles, n_bins))  # Initialize with zeros to accumulate accuracy across folds
    weights = np.full((n_splits, n_neurons), np.nan)
    weights_null = np.full((n_splits, n_neurons, n_shuffles), np.nan)
    dv = np.full((n_trials, n_bins), np.nan)
    dv_null = np.full((n_trials, n_bins, n_shuffles), np.nan)
    # dv_null = np.zeros((n_trials, n_bins, n_shuffles))  # Initialize with zeros to accumulate accuracy across folds

    # Split trials into training and testing sets (each fold gets a unique test set to prevent overfitting)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):

        # print(f'Fold {fold}:')
        # print(f'Train: index={train_index}')
        # print(f'Test: index={test_index}')

        # Loop over each time within_bin
        for within_bin in range(n_bins):

            # Define train and testing set for the current time within_bin and fold
            X_train, X_test = X[train_index, within_bin], X[test_index, within_bin]
            y_train, y_test = y[train_index], y[test_index]

            # Apply z-scoring normalization across neurons and time bins (otherwise might not converge)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)  # Fit and transform on the training set
            X_test = scaler.transform(X_test)  # Only transform the test set using the same scaler

            # Train decoder (logistic regression) on the current time within_bin’s neural activity
            clf = LogisticRegression(class_weight='balanced')
            clf.fit(X_train, y_train)

            # Evaluate decoder
            y_pred = clf.predict(X_test)  # Predicts the stimulus category for test trials
            y_score = clf.decision_function(X_test)  # Decision variable per trial (continuous distance to hyperplane)
            # y_acc = accuracy_score(y_test, y_pred)  # Computes accuracy for each fold and time within_bin
            y_acc = (y_pred == y_test).astype(int)  # Accuracy per trial

            # Store results
            pred[test_index, within_bin] = y_pred  # Predicted stimulus condition for each test trial at each time within_bin
            pred_err[test_index, within_bin] = y_pred - y_test  # Difference between predicted and actual labels
            acc[test_index, within_bin] = y_acc  # Accuracy for each test trial at each time within_bin
            weights[fold, :] = clf.coef_[0]  # Store decoder weights for each fold and time within_bin
            dv[test_index, within_bin] = y_score  # Store decision variable for each test trial at each time within_bin

            # Compute null distribution by shuffling the y_test (faster)
            y_test_shuffled = y_test.values.copy()
            for i in range(n_shuffles):
                np.random.shuffle(y_test_shuffled)
                # acc_null[i, bin_train] = accuracy_score(y_test_shuffled, y_pred)
                acc_null[i, within_bin] = np.mean((y_pred == y_test_shuffled).astype(int))

        # # Compute null distribution by shuffling y_train and fitting a null model (slower but best)
        # for i in tqdm(range(n_shuffles), desc=f'Fold {fold}/{n_splits}'):
        #     y_train_shuffled = np.random.permutation(y_train)  # Shuffle training labels
        #     clf_null = LogisticRegression(class_weight='balanced')  # Fit null model
        #     clf_null.fit(X_train, y_train_shuffled)
        #     weights_null[fold, :, i] = clf_null.coef_[0]  # Store null decoder weights for each fold and shuffle
        #
        #     # Loop over each time within_bin
        #     for within_bin in range(n_bins):
        #         X_test = X[test_index, within_bin]
        #         X_test = scaler.transform(X_test)
        #         y_pred_null = clf_null.predict(X_test)
        #         y_score_null = clf_null.decision_function(X_test)
        #         # acc_null[i, within_bin] = np.mean((y_pred_null == y_test).astype(int))  # Store null accuracy per trial
        #         # Overwrite each fold?
        #         acc_null[i, within_bin] += np.mean((y_pred_null == y_test).astype(int)) / n_splits
        #         # dv_null[test_index, within_bin, i] = y_score_null  # Store null decision variable for each test trial at each time within_bin
        #         # Overwrite each fold?
        #         dv_null[test_index, within_bin, i] += y_score_null / n_splits

    return pred, pred_err, acc, acc_null, weights, weights_null, dv, dv_null


# Mix of within decoder and the epoch split generalize decoder, but only storing the results of the generalization
@timer
def within_decoder_generalize(split, X=np.empty((1, 1, 1)), y=np.empty((1, 1)), n_shuffles=100):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_splits = skf.get_n_splits()

    n_trials, n_bins, n_neurons = X.shape
    index1 = np.where(split == 1)[0]
    index0 = np.where(split == 0)[0]
    n_trials0 = len(index0)

    pred = np.full((n_trials0, n_bins), np.nan)
    pred_err = np.full((n_trials0, n_bins), np.nan)
    acc = np.full((n_trials0, n_bins), np.nan)
    acc_null = np.full((n_shuffles, n_bins), np.nan)
    weights = np.full((n_splits, n_neurons), np.nan)
    weights_null = np.full((n_splits, n_neurons, n_shuffles), np.nan)
    dv = np.full((n_trials0, n_bins), np.nan)
    dv_null = np.full((n_trials0, n_bins, n_shuffles), np.nan)

    for fold, (train_index, _) in enumerate(skf.split(X[index1], y[index1])):
        train_index = index1[train_index]
        for within_bin in range(n_bins):
            X_train = X[train_index, within_bin]
            y_train = y[train_index]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            clf = LogisticRegression(class_weight='balanced')
            clf.fit(X_train, y_train)
            weights[fold, :] = clf.coef_[0]

            # ---- TEST ON SPLIT == 0 ----
            X_test = X[index0, within_bin]
            y_test = y[index0]
            X_test = scaler.transform(X_test)
            y_pred = clf.predict(X_test)
            y_score = clf.decision_function(X_test)
            y_acc = (y_pred == y_test).astype(int)
            pred[:, within_bin] = y_pred
            pred_err[:, within_bin] = y_pred - y_test
            acc[:, within_bin] = y_acc
            dv[:, within_bin] = y_score

            y_test_shuffled = y_test.values.copy()
            for i in range(n_shuffles):
                np.random.shuffle(y_test_shuffled)
                acc_null[i, within_bin] = np.mean((y_pred == y_test_shuffled).astype(int))

    return pred, pred_err, acc, acc_null, weights, weights_null, dv, dv_null


@timer
def cross_decoder(X=np.empty((1, 1, 1)), y=np.empty((1, 1)), n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary condition from neural data using K-fold cross-validation.
    Train in one bin and test in the rest, for each time bin.
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary stimulus condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    # Cross-validate results
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Stratified cross-validation
    n_splits = skf.get_n_splits()

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape
    pred = np.full((n_trials, n_bins, n_bins), np.nan)
    pred_err = np.full((n_trials, n_bins, n_bins), np.nan)
    acc = np.full((n_trials, n_bins, n_bins), np.nan)
    acc_null = np.full((n_shuffles, n_bins, n_bins), np.nan)
    # acc_null = np.zeros((n_shuffles, n_bins, n_bins))  # Initialize with zeros to accumulate accuracy across folds
    weights = np.full((n_splits, n_bins, n_neurons), np.nan)
    weights_null = np.full((n_splits, n_neurons, n_shuffles), np.nan)
    dv = np.full((n_trials, n_bins, n_bins), np.nan)
    dv_null = np.full((n_shuffles, n_bins, n_bins), np.nan)
    # dv_null = np.zeros((n_shuffles, n_bins, n_bins))  # Initialize with zeros to accumulate decision variable across folds

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Split trials into training and testing sets (each fold gets a unique test set to prevent overfitting)
    # for train_index, test_index in kf.split(X):
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):

        # print(f'Fold {fold}:')
        # print(f'Train: index={train_index}')
        # print(f'Test: index={test_index}')

        # Cross thinghy happens here
        for bin_train in range(n_bins):

            X_train = X[train_index, bin_train]
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
                y_score = clf.decision_function(X_test)  # Decision variable per trial (continuous distance to hyperplane)
                y_acc = accuracy_score(y_test, y_pred)  # Computes accuracy for each fold & time bin

                # Store results
                pred[test_index, bin_train, bin_test] = y_pred  # Predicted stimulus condition for each test trial at
                # each time bin
                pred_err[test_index, bin_train, bin_test] = y_pred - y_test  # Difference between predicted and labels
                acc[test_index, bin_train, bin_test] = y_acc  # Accuracy for each test trial at each time bin
                weights[fold, bin_train, :] = clf.coef_[0]  # Store decoder weights for each fold and time bin
                dv[test_index, bin_train, bin_test] = y_score  # Store decision variable for each test trial at each time bin

                # Compute null distribution by shuffling the labels and evaluating accuracy
                y_test_shuffled = y_test.values.copy()
                for i in range(n_shuffles):
                    np.random.shuffle(y_test_shuffled)
                    acc_null[i, bin_train, bin_test] = accuracy_score(y_test_shuffled, y_pred)

            # # Compute null distribution by shuffling y_train and fitting a null model (slower but best)
            # for i in tqdm(range(n_shuffles)):
            #     y_train_shuffled = np.random.permutation(y_train)  # Shuffle training labels
            #     clf_null = LogisticRegression(class_weight='balanced')  # Fit null model
            #     clf_null.fit(X_train, y_train_shuffled)
            #     weights_null[fold, :, i] = clf_null.coef_[0]  # Store null decoder weights for each fold and shuffle
            #
            #     # Evaluate null decoder on all test bins
            #     for bin_test in range(n_bins):
            #         X_test = X[test_index, bin_test]
            #         X_test = scaler.transform(X_test)
            #         y_pred_null = clf_null.predict(X_test)
            #         y_score_null = clf_null.decision_function(X_test)
            #         acc_null[i, bin_train, bin_test] = accuracy_score(y_test, y_pred_null)  # Overwrite each fold?
            #         # acc_null[i, bin_train, bin_test] += accuracy_score(y_test, y_pred_null) / n_splits
            #         dv_null[i, bin_train, bin_test] = np.mean(y_score_null)  # Overwrite each fold?
            #         # dv_null[i, bin_train, bin_test] += np.mean(y_score_null) / n_splits

    return pred, pred_err, acc, acc_null, weights, weights_null, dv, dv_null


@timer
def epoch_cross_decoder(bins, epoch=None, X=np.empty((1, 1, 1)), y=np.empty((1, 1)), n_shuffles=100):
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

    # Cross-validate results
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Stratified cross-validation
    n_splits = skf.get_n_splits()

    # Initialize arrays
    # 1 less dimension (bins) than cross decoder, as analyzing only on bin (or the average of some)
    n_trials, n_bins, n_neurons = X.shape
    pred = np.full((n_trials, n_bins), np.nan)
    pred_err = np.full((n_trials, n_bins), np.nan)
    acc = np.full((n_trials, n_bins), np.nan)
    acc_null = np.full((n_trials, n_bins, n_shuffles), np.nan)
    weights = np.full((n_splits, n_neurons), np.nan)
    weights_null = np.full((n_splits, n_neurons, n_shuffles), np.nan)
    dv = np.full((n_trials, n_bins), np.nan)
    dv_null = np.full((n_trials, n_bins, n_shuffles), np.nan)

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Define epoch of interest
    epoch_targets = {
        'stim': (0.0, 0.1),  # Align to stimulus onset
        'delay': (0.9, 1.0),  # Align to stimulus onset
        'resp': (1.9, 2.0),  # Align to stimulus onset
        'first_lick': (-0.05, 0.0),  # Align to response (first lick)
        'mid_lick': (0.5, 0.55),  # Align to response (first lick)
        'late_lick': (0.95, 1.0),  # Align to response (first lick)
        'post_lick': (1.5, 1.55)  # Align to response (first lick)
    }

    # Compute start/end indices for the requested epoch
    if epoch not in epoch_targets:
        raise ValueError("Epoch must be one of: 'stim', 'delay', 'resp', 'first_lick', 'mid_lick', 'late_lick', 'post_lick'.")

    start_time, end_time = epoch_targets[epoch]
    bin_size = bins[1] - bins[0]
    epoch_start_idx = round((start_time - bins[0]) / bin_size)
    epoch_end_idx = round((end_time - bins[0]) / bin_size)

    # Split trials into training and testing sets (each fold gets a unique test set to prevent overfitting)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):

        # print(f'Fold {fold}:')
        # print(f'Train: index={train_index}')
        # print(f'Test: index={test_index}')

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
            y_score = clf.decision_function(X_test)  # Decision variable per trial (continuous distance to hyperplane)
            # y_acc = accuracy_score(y_test, y_pred)  # Computes accuracy for each fold
            y_acc = (y_pred == y_test).astype(int)  # Computes accuracy per trial

            # Store results
            pred[test_index, bin_test] = y_pred  # Predicted stimulus condition for each test trial at each time bin
            pred_err[test_index, bin_test] = y_pred - y_test  # Difference between predicted and actual labels
            # acc[fold, bin_test] = y_acc  # Accuracy for fold at each time bin
            acc[test_index, bin_test] = y_acc  # Accuracy for each test trial at each time bin
            weights[fold, :] = clf.coef_[0]  # Store decoder weights for each fold
            dv[test_index, bin_test] = y_score  # Store decision variable for each test trial at each time bin

            y_test_shuffled = y_test.values.copy()
            for _ in range(n_shuffles):
                np.random.shuffle(y_test_shuffled)
                # acc_null[_, bin_test] = accuracy_score(y_test_shuffled, y_pred)
                acc_null[test_index, bin_test, _] = (y_pred == y_test_shuffled).astype(int)

        # # Compute null distribution by shuffling y_train and fitting a null model (slower but best)
        # for i in tqdm(range(n_shuffles)):
        #     y_train_shuffled = np.random.permutation(y_train)  # Shuffle training labels
        #     clf_null = LogisticRegression(class_weight='balanced')  # Fit null model
        #     clf_null.fit(X_train, y_train_shuffled)
        #     weights_null[fold, :, i] = clf_null.coef_[0]  # Store null decoder weights for each fold and shuffle
        #
        #     # Loop over each time bin_train
        #     for bin_test in range(n_bins):
        #         X_test = X[test_index, bin_test]
        #         X_test = scaler.transform(X_test)
        #         y_pred_null = clf_null.predict(X_test)
        #         y_score_null = clf_null.decision_function(X_test)
        #         acc_null[test_index, bin_test, i] = (y_pred_null == y_test).astype(int)  # Store null accuracy per trial
        #         dv_null[test_index, bin_test, i] = y_score_null  # Store null decision variable for each test trial at each time bin

    return pred, pred_err, acc, acc_null, weights, weights_null, dv, dv_null


@timer
def epoch_cross_decoder_split(bins, split, epoch=None, X=np.empty((1, 1, 1)), y=np.empty((1, 1)), n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary condition from neural data using K-fold cross-validation.
    Train in one bin and test in the rest, for one epoch (one bin or the mean of a few). Akin of takin a slice of the
    cross-decoder matrix.
    Train in all trials and test separately for each condition.
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    # Cross-validate results
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_splits = skf.get_n_splits()

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape
    pred = [np.full((n_trials, n_bins), np.nan), np.full((n_trials, n_bins), np.nan)]
    pred_err = [np.full((n_trials, n_bins), np.nan), np.full((n_trials, n_bins), np.nan)]
    acc = [np.full((n_trials, n_bins), np.nan), np.full((n_trials, n_bins), np.nan)]
    acc_null = [np.full((n_trials, n_bins, n_shuffles), np.nan), np.full((n_trials, n_bins, n_shuffles), np.nan)]
    weights = [np.full((n_splits, n_neurons), np.nan), np.full((n_splits, n_neurons), np.nan)]
    weights_null = [np.full((n_splits, n_neurons, n_shuffles), np.nan), np.full((n_splits, n_neurons, n_shuffles), np.nan)]
    dv = [np.full((n_trials, n_bins), np.nan), np.full((n_trials, n_bins), np.nan)]
    dv_null = [np.full((n_trials, n_bins, n_shuffles), np.nan), np.full((n_trials, n_bins, n_shuffles), np.nan)]

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Define epoch of interest
    epoch_targets = {
        'stim': (0.0, 0.1),  # Align to stimulus onset
        'delay': (0.9, 1.0),  # Align to stimulus onset
        'resp': (1.9, 2.0),  # Align to stimulus onset
        'first_lick': (-0.05, 0.0),  # Align to response (first lick)
        'mid_lick': (0.5, 0.55),  # Align to response (first lick)
        'late_lick': (0.95, 1.0),  # Align to response (first lick)
        'post_lick': (1.5, 1.55)  # Align to response (first lick)
    }

    # Compute start/end indices for the requested epoch
    if epoch not in epoch_targets:
        raise ValueError("Epoch must be one of: 'stim', 'delay', 'resp', 'first_lick', 'mid_lick', 'late_lick', 'post_lick'.")

    start_time, end_time = epoch_targets[epoch]
    bin_size = bins[1] - bins[0]
    epoch_start_idx = round((start_time - bins[0]) / bin_size)
    epoch_end_idx = round((end_time - bins[0]) / bin_size)

    # Split trials into training and testing sets (each fold gets a unique test set to prevent over-fitting)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):

        # print(f'Fold {fold}:')
        # print(f'Train: index={train_index}')
        # print(f'Test: index={test_index}')

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
            y_score0 = clf.decision_function(X_test0)  # Decision variable per trial (continuous distance to hyperplane)
            # y_acc0 = accuracy_score(y_test0, y_pred0)  # Computes accuracy for each fold & time bin_train
            y_acc0 = (y_pred0 == y_test0).astype(int)  # Computes accuracy per trial
            # print(f"Accuracy: {y_acc0:.2f}")
            y_pred1 = clf.predict(X_test1)  # Predicts the stimulus category for test trials
            y_score1 = clf.decision_function(X_test1)  # Decision variable per trial (continuous distance to hyperplane)
            # y_acc1 = accuracy_score(y_test1, y_pred1)  # Computes accuracy for each fold & time bin_train
            y_acc1 = (y_pred1 == y_test1).astype(int)  # Computes accuracy per trial
            # print(f"Accuracy: {y_acc1:.2f}")

            # Store results
            pred[0][test_idx0, bin_test] = y_pred0  # Predicted stimulus condition for each test trial at each time bin
            pred_err[0][test_idx0, bin_test] = y_pred0 - y_test0  # Difference between predicted and actual labels
            # acc0[fold, bin_test] = y_acc0  # Accuracy for each test trial at each time bin
            acc[0][test_idx0, bin_test] = y_acc0  # Accuracy per trial
            weights[0][fold, :] = clf.coef_[0]  # Store decoder weights for each fold
            pred[1][test_idx1, bin_test] = y_pred1  # Predicted stimulus condition for each test trial at each time bin
            pred_err[1][test_idx1, bin_test] = y_pred1 - y_test1  # Difference between predicted and actual labels
            # acc1[fold, bin_test] = y_acc1  # Accuracy for each test trial at each time bin
            acc[1][test_idx1, bin_test] = y_acc1  # Accuracy per trial
            weights[1][fold, :] = clf.coef_[0]  # Store decoder weights for each fold
            dv[0][test_idx0, bin_test] = y_score0  # Store decision variable for each test trial at each time bin
            dv[1][test_idx1, bin_test] = y_score1  # Store decision variable for each test trial at each time bin

            # Compute null distribution by shuffling the labels and evaluating accuracy
            y_test_shuffled0 = y_test0.values.copy()
            y_test_shuffled1 = y_test1.values.copy()
            for i in range(n_shuffles):
                np.random.shuffle(y_test_shuffled0)
                # acc_null0[fold, i, bin_test] = accuracy_score(y_test_shuffled0, y_pred0)
                acc_null[0][test_idx0, bin_test, i] = (y_pred0 == y_test_shuffled0).astype(int)  # Accuracy per trial for
                # shuffled labels
                # pred_err_temp.append(y_pred - y_test_shuffled)
                # pred_err_null0[test_idx0, bin_test, i] = y_pred0 - y_test_shuffled0  # Difference between predicted and
                # actual labels

                np.random.shuffle(y_test_shuffled1)
                # acc_null1[fold, i, bin_test] = accuracy_score(y_test_shuffled1, y_pred1)
                acc_null[1][test_idx1, bin_test, i] = (y_pred1 == y_test_shuffled1).astype(int)  # Accuracy per trial for
                # shuffled labels
                # pred_err_temp.append(y_pred - y_test_shuffled)
                # pred_err_null1[test_idx1, bin_test, i] = y_pred1 - y_test_shuffled1  # Difference between predicted and
                # actual labels

                # # TEST FOR NON FLAT SHUFFLES
                # y_train_shuffled = np.random.permutation(y_train)
                # clf_null = LogisticRegression(class_weight='balanced')
                # clf_null.fit(X_train, y_train_shuffled)
                #
                # # Predict on the same test sets
                # y_pred0_null = clf_null.predict(X_test0)
                # y_pred1_null = clf_null.predict(X_test1)
                #
                # # Compute per-trial accuracy
                # acc_null[0][test_idx0, bin_test, i] = (y_pred0_null == y_test0).astype(int)
                # acc_null[1][test_idx1, bin_test, i] = (y_pred1_null == y_test1).astype(int)

    return pred, pred_err, acc, acc_null, weights, weights_null, dv, dv_null


@timer
def epoch_cross_decoder_generalize(bins, split, epoch=None, X=np.empty((1, 1, 1)), y=np.empty((1, 1)), n_shuffles=100):
    """
    Perform logistic regression-based decoding of a binary condition from neural data using K-fold cross-validation.
    Train in one bin and test in the rest, for one epoch (one bin or the mean of a few). Akin of takin a slice of the
    cross-decoder matrix.
    Train in one condition and test in that condition (cross-validation) and in the other condition (generalization).
    :param X: 3D array with neural data (trials x time x neurons)
    :param y: 1D array with binary condition
    :return: pred, pred_err (predicted condition and prediction error)
    """

    # Cross-validate results
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_splits = skf.get_n_splits()

    # Initialize arrays
    n_trials, n_bins, n_neurons = X.shape
    pred = [np.full((n_trials, n_bins), np.nan), np.full((n_trials, n_bins), np.nan)]
    pred_err = [np.full((n_trials, n_bins), np.nan), np.full((n_trials, n_bins), np.nan)]
    acc = [np.full((n_trials, n_bins), np.nan), np.full((n_trials, n_bins), np.nan)]
    acc_null = [np.full((n_trials, n_bins, n_shuffles), np.nan), np.full((n_trials, n_bins, n_shuffles), np.nan)]
    weights = [np.full((n_splits, n_neurons), np.nan), np.full((n_splits, n_neurons), np.nan)]
    weights_null = [np.full((n_splits, n_neurons, n_shuffles), np.nan), np.full((n_splits, n_neurons, n_shuffles), np.nan)]
    dv = [np.full((n_trials, n_bins), np.nan), np.full((n_trials, n_bins), np.nan)]
    dv_null = [np.full((n_trials, n_bins, n_shuffles), np.nan), np.full((n_trials, n_bins, n_shuffles), np.nan)]

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Define epoch of interest
    epoch_targets = {
        'stim': (0.0, 0.1),  # Align to stimulus onset
        'delay': (0.9, 1.0),  # Align to stimulus onset
        'resp': (1.9, 2.0),  # Align to stimulus onset
        'first_lick': (-0.05, 0.0),  # Align to response (first lick)
        'mid_lick': (0.5, 0.55),  # Align to response (first lick)
        'late_lick': (0.95, 1.0),  # Align to response (first lick)
        'post_lick': (1.5, 1.55)  # Align to response (first lick)
    }

    # Compute start/end indices for the requested epoch
    if epoch not in epoch_targets:
        raise ValueError("Epoch must be one of: 'stim', 'delay', 'resp', 'first_lick', 'mid_lick', 'late_lick', 'post_lick'.")

    start_time, end_time = epoch_targets[epoch]
    bin_size = bins[1] - bins[0]
    epoch_start_idx = round((start_time - bins[0]) / bin_size)
    epoch_end_idx = round((end_time - bins[0]) / bin_size)

    index1 = np.where(split == 1)[0]  # Indices of correct/engaged trials
    index0 = np.where(split == 0)[0]  # Indices of error/disengaged trials (never used for training)

    # Split trials into training and testing sets (each fold gets a unique test set to prevent over-fitting)
    for fold, (train_index, test_index) in enumerate(skf.split(X[index1], y[index1])):

        # print(f'Fold {fold}:')
        # print(f'Train: index={train_index}')
        # print(f'Test: index={test_index}')

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
            y_score0 = clf.decision_function(X_test0)  # Decision variable per trial (continuous distance to hyperplane)
            # y_acc0 = accuracy_score(y_test0, y_pred0)  # Computes accuracy for each fold & time bin_train
            y_acc0 = (y_pred0 == y_test0).astype(int)  # Computes accuracy per trial
            # print(f"Accuracy: {y_acc0:.2f}")
            y_pred1 = clf.predict(X_test1)  # Predicts the stimulus category for test trials
            y_score1 = clf.decision_function(X_test1)  # Decision variable per trial (continuous distance to hyperplane)
            # y_acc1 = accuracy_score(y_test1, y_pred1)  # Computes accuracy for each fold & time bin_train
            y_acc1 = (y_pred1 == y_test1).astype(int)  # Computes accuracy per trial
            # print(f"Accuracy: {y_acc1:.2f}")

            # Store results
            pred[0][test_idx0, bin_test] = y_pred0  # Predicted stimulus condition for each test trial at each time bin
            pred_err[0][test_idx0, bin_test] = y_pred0 - y_test0  # Difference between predicted and actual labels
            # acc0[fold, bin_test] = y_acc0  # Accuracy for each test trial at each time bin
            acc[0][test_idx0, bin_test] = y_acc0  # Accuracy per trial
            weights[0][fold, :] = clf.coef_[0]  # Store decoder weights for each fold
            pred[1][test_idx1, bin_test] = y_pred1  # Predicted stimulus condition for each test trial at each time bin
            pred_err[1][test_idx1, bin_test] = y_pred1 - y_test1  # Difference between predicted and actual labels
            # acc1[fold, bin_test] = y_acc1  # Accuracy for each test trial at each time bin
            acc[1][test_idx1, bin_test] = y_acc1  # Accuracy per trial
            weights[0][fold, :] = clf.coef_[0]  # Store decoder weights for each fold
            dv[0][test_idx0, bin_test] = y_score0  # Store decision variable for each test trial at each time bin
            dv[1][test_idx1, bin_test] = y_score1  # Store decision variable for each test trial at each time bin

            # Compute null distribution by shuffling the labels and evaluating accuracy
            y_test_shuffled0 = y_test0.values.copy()
            y_test_shuffled1 = y_test1.values.copy()
            for i in range(n_shuffles):
                np.random.shuffle(y_test_shuffled0)
                # acc_null0[fold, i, bin_test] = accuracy_score(y_test_shuffled0, y_pred0)
                acc_null[0][test_idx0, bin_test, i] = (y_pred0 == y_test_shuffled0).astype(int)  # Accuracy per trial for
                # shuffled labels
                # pred_err_temp.append(y_pred - y_test_shuffled)
                # pred_err_null0[test_idx0, bin_test, i] = y_pred0 - y_test_shuffled0  # Difference between predicted and
                # actual labels

                np.random.shuffle(y_test_shuffled1)
                # acc_null1[fold, i, bin_test] = accuracy_score(y_test_shuffled1, y_pred1)
                acc_null[1][test_idx1, bin_test, i] = (y_pred1 == y_test_shuffled1).astype(int)  # Accuracy per trial for
                # shuffled labels
                # pred_err_temp.append(y_pred - y_test_shuffled)
                # pred_err_null1[test_idx1, bin_test, i] = y_pred1 - y_test_shuffled1  # Difference between predicted and
                # actual labels

                # # TEST FOR NON FLAT SHUFFLES
                # y_train_shuffled = np.random.permutation(y_train)
                # clf_null = LogisticRegression(class_weight='balanced')
                # clf_null.fit(X_train, y_train_shuffled)
                #
                # # Predict on the same test sets
                # y_pred0_null = clf_null.predict(X_test0)
                # y_pred1_null = clf_null.predict(X_test1)
                #
                # # Compute per-trial accuracy
                # acc_null[0][test_idx0, bin_test, i] = (y_pred0_null == y_test0).astype(int)
                # acc_null[1][test_idx1, bin_test, i] = (y_pred1_null == y_test1).astype(int)

    return pred, pred_err, acc, acc_null, weights, weights_null, dv, dv_null


@timer
def mean_decoder(subject, what='stim', align='stim', kind=None, epoch=None, epoch_ortho=None, split_by=None,
                 drop_miss=True, hit_only=False, engagement=None, group=None, depth=None, min_fr=None, max_fano=None,
                 n_shuffles=100, save=False):
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
    :param group: 'good' for isolated units, 'mua' for multiunit activity, None for both
    :param depth: tuple with min and max depth to include (surface = 0 mm, max depth = 4 mm)
    :param n_shuffles: number of shuffles to perform
    :param save: whether to save the results as a pickle
    :return: results (dict)
    """

    if what == 'stim':
        col = 'Side'
    elif what == 'choice':
        col = 'Choice'
    elif what == 'prev_choice':
        col = 'PrevChoice'
    else:
        raise ValueError("'what' (to decode) must be 'stim', 'choice' or 'prev_choice'")

    results = {
        'pred': [],
        'pred_err': [],
        'acc': [],
        'acc_null': [],
        'bins': [],
        'weights': [],
        'weights_null': [],
        'dv': [],
        'dv_null': [],
        'df': []
    }

    folder_parent = Path.home() / 'data' / subject
    ephys_ids = get_ephys_sessions(subject)
    # error_sessions = preprocess_subject(subject)
    # ephys_ids = [id for id in ephys_ids if id not in error_sessions]

    error_sessions = []
    filename = None

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

            # Filter units
            # path_cluster_info = Path('/archive/mouse/Alexis ephys/spike_sorting') / subject / ephys_ids[i] / 'phy2'
            path_cluster_info = Path('/archive/alexis/ephys/spike_sorting') / subject / ephys_ids[i] / 'phy2'
            cluster_info = pd.read_csv(path_cluster_info / 'cluster_info.tsv', sep='\t')
            cluster_info = cluster_info[cluster_info['group'] != 'noise'].reset_index(drop=True)  # Drop noise units
            cluster_info = cluster_info.loc[cluster_info.group.isin(['good', 'mua'])].reset_index(drop=True) # Keep good/mua only
            assert all_psth.shape[2] == len(cluster_info)
            all_psth, cluster_info = filter_units(bins, all_psth, cluster_info, min_fr=min_fr, max_fano=max_fano,
                                                  group=group, depth=depth)

            # Load GLM-HMM fit
            # path_glmhmm = ((Path.home() / 'PycharmProjects' / 'glmhmm' / 'TEST' / '2s_3cov' / '2AFC_5' / subject)
            #                .with_suffix('.csv'))  # Fitted all sessions (non-ephys and ephys)
            path_glmhmm = ((Path.home() / 'PycharmProjects' / 'glmhmm' / 'TEST' / '2s_3cov' / 'Ephys' / subject)
                           .with_suffix('.csv'))  # Fitted sphys sessions only
            df_glmhmm = pd.read_csv(path_glmhmm)  # All sessions concatenated
            session = df_behavior.Session.unique()[0]
            df_glmhmm = df_glmhmm[df_glmhmm.Session == session].reset_index(drop=True)

            # Drop misses (GLM-HMM lack misses)
            resp_idx = df_behavior[df_behavior.Miss == 0].index
            df_behavior = df_behavior.iloc[resp_idx].reset_index(drop=True)
            all_psth = all_psth[resp_idx]
            assert len(df_glmhmm) == len(df_behavior)  # Check sessions match
            df_behavior = df_glmhmm
            drop_miss = False  # To not do it twice

            # Drop misses in previous choices (NaN in AfterHit)
            if what == 'prev_choice':
                df_behavior['PrevChoice'] = df_behavior.groupby('Session')['Choice'].shift(1)
                resp_idx = df_behavior[df_behavior['PrevChoice'].notna()].index
                df_behavior = df_behavior.iloc[resp_idx].reset_index(drop=True)
                all_psth = all_psth[resp_idx]

            state_label = ''
            if engagement is not None:  # Add engaged column to df_behavior
                # Filter by engagement
                mask = (df_behavior.State == engagement).to_numpy()
                df_behavior = df_behavior[mask].reset_index(drop=True)
                all_psth = all_psth[mask]
                state_label = '_engaged' if engagement == 1 else '_disengaged'

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
                pred, pred_err, acc, acc_null, weights, weights_null, dv, dv_null = \
                    within_decoder(all_psth, df_behavior[col], n_shuffles)
                filename = f'{what}_{kind}_decoder{hit_label}{state_label}_({align}_aligned).pkl'
            elif kind == 'cross':
                pred, pred_err, acc, acc_null, weights, weights_null, dv, dv_null = \
                    cross_decoder(all_psth, df_behavior[col], n_shuffles)
                filename = f'{what}_{kind}{hit_label}{state_label}_({align}_aligned).pkl'
            elif kind == 'epoch':
                pred, pred_err, acc, acc_null, weights, weights_null, dv, dv_null = \
                    epoch_cross_decoder(bins, epoch, all_psth, df_behavior[col], n_shuffles)
                filename = f'{what}_{kind}_decoder_{epoch}{hit_label}{state_label}_({align}_aligned).pkl'
            elif kind == 'epoch_split':
                split = df_behavior[split_by]
                pred, pred_err, acc, acc_null, weights, weights_null, dv, dv_null = \
                    epoch_cross_decoder_split(bins, split, epoch, all_psth, df_behavior[col], n_shuffles)
                filename = f'{what}_{kind}_decoder_{epoch}_split_by_{split_by}{hit_label}{state_label}_({align}_aligned).pkl'
            elif kind == 'generalize':
                split = df_behavior[split_by]
                pred, pred_err, acc, acc_null, weights, weights_null, dv, dv_null = \
                    epoch_cross_decoder_generalize(bins, split, epoch, all_psth, df_behavior[col], n_shuffles)
                filename = f'{what}_{kind}_decoder_{epoch}_generalize_{split_by}{hit_label}{state_label}_({align}_aligned).pkl'

            # Under construction (use at your own risk)
            elif kind == 'epoch_ortho':
                pred, pred_err, acc, acc_null = \
                    epoch_cross_decoder_ORTHO(bins, epoch, epoch_ortho, all_psth, df_behavior[col], n_shuffles)
                filename = f'epoch_cross_decoder_ORTHO_{epoch}_{state_label}.pkl'
            elif kind == 'within_generalize':
                split = df_behavior[split_by]
                pred, pred_err, acc, acc_null, weights, weights_null, dv, dv_null = \
                    within_decoder_generalize(split, all_psth, df_behavior[col], n_shuffles)
                filename = f'{what}_{kind}_decoder{hit_label}{state_label}_({align}_aligned).pkl'
            else:
                raise ValueError("Kind must be 'within', 'within_generalize', 'cross', 'epoch', 'epoch_split' or 'generalize'")

            results['pred'].append(pred)
            results['pred_err'].append(pred_err)
            results['acc'].append(acc)
            results['acc_null'].append(acc_null)
            results['weights'].append(weights)
            results['weights_null'].append(weights_null)
            results['dv'].append(dv)
            results['dv_null'].append(dv_null)
            results['bins'] = bins
            results['df'].append(df_behavior)

        except Exception as e:
            print(f'An error occurred in session {ephys_ids[i]}: {e}')
            error_sessions.append(ephys_ids[i])
            traceback.print_exc()
            continue

    if save:
        if filename:
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


def get_selective_units(weights, weights_null, alpha=0.05):
    """
    Identify selective neurons from decoder by comparing real weights to null distribution.
    :param weights: 2D array (folds x neurons)
    :param weights_null: 3D array (folds x neurons x shuffles)
    :param alpha: significance level (default 0.05)
    :return: boolean array of shape (neurons) indicating selective neurons
    """

    # Mean weight (shape: neurons)
    mean_weights = weights.mean(axis=0)

    # Mean null distribution per neuron across folds (shape: neurons, shuffles)
    weights_null = weights_null.mean(axis=0)

    # Compute percentiles
    lower = np.percentile(weights_null, 100 * (alpha / 2), axis=1)
    upper = np.percentile(weights_null, 100 * (1 - alpha / 2), axis=1)

    # Selective neurons: real mean outside null interval
    selective_mask = (mean_weights < lower) | (mean_weights > upper)

    return selective_mask


def get_selectivity(bins, X, y, epoch='stim', alpha=0.05, n_shuffles=100):
    """
    Compute neuron-by-neuron Poisson GLM for a given epoch to estimate selectivity.
    :param bins: 1D array with time bins
    :param epoch: epoch of interest (str: 'stim', 'delay', 'resp')
    :param X: 3D array with neural data (trials x time x neurons)
    :param Y: 1D array with binary stimulus condition
    :param n_shuffles: number of shuffles to perform
    :return: params, pvalues, boolean mask for selective_neurons
    """

    # Define epoch of interest
    epoch_targets = {
        'stim': (0.0, 0.1),  # Align to stimulus onset
        'delay': (0.9, 1.0),  # Align to stimulus onset
        'resp': (1.9, 2.0),  # Align to stimulus onset
        'first_lick': (-0.05, 0.0),  # Align to response (first lick)
        'mid_lick': (0.5, 0.55),  # Align to response (first lick)
        'late_lick': (0.95, 1.0),  # Align to response (first lick)
        'post_lick': (1.5, 1.55)  # Align to response (first lick)
    }

    # Compute start/end indices for the requested epoch
    if epoch not in epoch_targets:
        raise ValueError("Epoch must be one of: 'stim', 'delay', 'resp', 'first_lick', 'mid_lick', 'late_lick', 'post_lick'.")

    start_time, end_time = epoch_targets[epoch]
    bin_size = bins[1] - bins[0]
    epoch_start_idx = round((start_time - bins[0]) / bin_size)
    epoch_end_idx = round((end_time - bins[0]) / bin_size)

    n_trials, n_bins, n_neurons = X.shape

    params = np.zeros(n_neurons)
    pvalues = np.ones(n_neurons)

    # Precompute spike counts per neuron for the epoch
    X = X[:, epoch_start_idx:epoch_end_idx, :].sum(axis=1)  # trials x neurons

    for neuron in range(n_neurons):
        X_neuron = X[:, neuron]

        # Skip neurons with no spikes or no variance
        if np.all(X_neuron == 0) or np.var(X_neuron) == 0:
            params[neuron] = 0
            pvalues[neuron] = 1
            continue

        # Fit GLM on real data
        model = sm.GLM(X_neuron, sm.add_constant(y), family=sm.families.Poisson())
        results = model.fit()
        param = results.params[1]
        params[neuron] = param

        # Compute p-value
        if n_shuffles > 0:
            null_params = np.zeros(n_shuffles)
            for i in range(n_shuffles):
                y_shuffled = np.random.permutation(y)
                # model_null = sm.GLM(y_shuffled, X_neuron, family=sm.families.Poisson())
                model_null = sm.GLM(X_neuron, sm.add_constant(y_shuffled), family=sm.families.Poisson())
                results_null = model_null.fit()
                null_params[i] = results_null.params[1]
            pvalues[neuron] = np.mean(np.abs(null_params) >= np.abs(param))
        else:
            pvalues[neuron] = results.pvalues[1]

    selective_neurons = np.where(pvalues < alpha)[0]
    selective_mask = pvalues < alpha
    print(f'{epoch} selectivity: {len(selective_neurons)/n_neurons*100:.1f}% ({len(selective_neurons)}/{n_neurons} neurons)')

    return params, pvalues, selective_mask


def mean_selectivity(subject, what='stim', align='stim', epoch=None, group=None, depth=None, min_fr=None, max_fano=None,
                     alpha=0.05, n_shuffles=0, save=False):
    """
    Perform within time bin decoder across all sessions for one subject.
    :param subject: subject ID (str)
    :param what: what to decode ('stim' or 'choice'). If correct trials only, they are the same ('choice'='stim')
    :param align: alignment of neural data ('stim', or 'resp')
    :param epoch: epoch of interest (str: 'stim', 'delay', 'resp')
    :param group: 'good' for isolated units, 'mua' for multiunit activity, None for both
    :param depth: tuple with min and max depth to include (surface = 0 mm, max depth = 4 mm)
    :param n_shuffles: number of shuffles to perform
    :param save: whether to save the results as a pickle
    :return: results (dict)
    """

    if what == 'stim':
        col = 'Side'
    elif what == 'choice':
        col = 'Choice'
    else:
        raise ValueError("'what' (to decode) must be 'stim' or 'choice'")

    selectivity = {
        'params': [],
        'pvalues': [],
        'selective_mask': [],
        'cluster_info': []
    }

    folder_parent = Path.home() / 'data' / subject
    ephys_ids = get_ephys_sessions(subject)
    error_sessions = []

    for i in range(len(ephys_ids)):

        folder_child = folder_parent / ephys_ids[i]

        try:
            print(f'Processing session {i + 1}/{len(ephys_ids)}: {ephys_ids[i]}...')
            filename_behavior = [f for f in os.listdir(folder_child) if f.endswith('.csv')][0]  # Assume only one .csv file
            path_behavior = folder_child / filename_behavior
            df_behavior = pd.read_csv(path_behavior)
            bins = np.load(folder_child / f'bins_{align}.npy')
            all_psth = np.load(folder_child / f'all_psth_{align}.npy')

            # Filter units
            # path_cluster_info = Path('/archive/mouse/Alexis ephys/spike_sorting') / subject / ephys_ids[i] / 'phy2'
            path_cluster_info = Path('/archive/alexis/ephys/spike_sorting') / subject / ephys_ids[i] / 'phy2'
            cluster_info = pd.read_csv(path_cluster_info / 'cluster_info.tsv', sep='\t')
            cluster_info = cluster_info[cluster_info['group'] != 'noise'].reset_index(drop=True)  # Drop noise units
            cluster_info = cluster_info.loc[cluster_info.group.isin(['good', 'mua'])].reset_index(drop=True) # Keep good/mua only
            surface = cluster_info.depth.max()
            cluster_info['depth'] = surface - cluster_info['depth']  # Depth from surface
            rec_side = get_rec_side(ephys_ids[i])
            cluster_info['RECside'] = rec_side
            assert all_psth.shape[2] == len(cluster_info)
            all_psth, cluster_info = filter_units(bins, all_psth, cluster_info, min_fr=min_fr, max_fano=max_fano,
                                                  group=group, depth=depth)

            # Load GLM-HMM fit
            path_glmhmm = ((Path.home() / 'PycharmProjects' / 'glmhmm' / 'TEST' / '2s_3cov' / 'Ephys' / subject)
                           .with_suffix('.csv'))  # Fitted sphys sessions only
            df_glmhmm = pd.read_csv(path_glmhmm)  # All sessions concatenated
            session = df_behavior.Session.unique()[0]
            df_glmhmm = df_glmhmm[df_glmhmm.Session == session].reset_index(drop=True)

            # Drop misses (GLM-HMM lack misses)
            resp_idx = df_behavior[df_behavior.Miss == 0].index
            df_behavior = df_behavior.iloc[resp_idx].reset_index(drop=True)
            all_psth = all_psth[resp_idx]
            assert len(df_glmhmm) == len(df_behavior)  # Check sessions match
            df_behavior = df_glmhmm

            # Filter only correct engaged trials
            mask = ((df_behavior.State == 1)&((df_behavior.Hit == 1))).to_numpy()
            df_behavior = df_behavior[mask].reset_index(drop=True)
            all_psth = all_psth[mask]

            params, pvalues, selective_mask = get_selectivity(bins, all_psth, df_behavior[col], epoch=epoch, alpha=alpha,
                                                              n_shuffles=n_shuffles)
            filename = f'{what}_selectivity_{epoch}_({align}_aligned).pkl'
            selectivity['params'].append(params)
            selectivity['pvalues'].append(pvalues)
            selectivity['selective_mask'].append(selective_mask)
            selectivity['cluster_info'].append(cluster_info)

        except Exception as e:
            print(f'An error occurred in session {ephys_ids[i]}: {e}')
            error_sessions.append(ephys_ids[i])
            traceback.print_exc()
            continue

    if save:
        if filename:
            os.chdir(folder_parent)
            with open(filename, 'wb') as f:
                pickle.dump(selectivity, f)

    return selectivity


def add_selectivity_cluster_info(mean_selectivity):

    cluster_info = pd.concat(mean_selectivity['cluster_info'], ignore_index=True)  # Unpack df
    weights, pvalues, selective_mask, n_selective = [], [], [], []  # Initialize lists

    for p, pv, m in zip(mean_selectivity['params'],
                    mean_selectivity['pvalues'],
                    mean_selectivity['selective_mask']):
        weights.append(p)
        pvalues.append(pv)
        selective_mask.append(m)
        n_selective.append(sum(m))

    weights = np.concatenate(weights)
    pvalues = np.concatenate(pvalues)
    selective_mask = np.concatenate(selective_mask)
    cluster_info['weights'] = weights
    cluster_info['pvalues'] = pvalues
    cluster_info['selective_mask'] = selective_mask

    return cluster_info


def lr2ic(weights, rec_side):
    """
    Convert weights from left/right to contra/ipsi.
    Assumes 0=left and 1=right when fitting the weights.
    Negative weights indicate contra preference, positive weights indicate ipsi preference.

    """
    weights = weights * (2 * rec_side - 1)
    return weights


########################################################################################################################
# PLOT DECODERS
########################################################################################################################


def plot_within_decoder(bins, acc, acc_null, z_null=False, excess=True):
    """
    Plot the decoding accuracy and the null distribution of accuracy.
    :param bins: 1D array with time bins
    :param acc: 2D array with decoding accuracy (trials x time)
    :param acc_null: 2D array with null distribution of accuracy (shuffles x time)
    :param z_null: whether to Z-score the decoding accuracy by the null distribution of accuracy
    :param excess: whether to plot excess accuracy (acc - null mean)
    """

    # Plot decoding accuracy
    # plt.figure(constrained_layout=True)
    plt.axvline(0, color='tab:gray', linestyle='-')  # Stimulus onset
    plt.axvline(1, color='tab:gray', linestyle='-')  # Go cue

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute p-values and z-scores relative to the null distribution
    if z_null:
        z_scores = null_zscore(acc, acc_null)
        p_values = p_val(acc, acc_null)
        significant_region = p_values < 0.05  # When assessing significance of single sessions use p < 0.05
        plt.plot(bin_centers, z_scores, color='k', label='Z acc.')
        plt.fill_between(bin_centers, z_scores, color='tab:gray', where=significant_region, edgecolor='none',
                         alpha=0.25, label='α < .05')
        plt.axhline(0, color='tab:gray', linestyle='--')  # Chance level
        ylabel = 'Z-score'
    else:
        acc_mean = acc.mean(axis=0)
        acc_sem = sem(acc, axis=0)
        acc_null_mean = acc_null.mean(axis=0)
        acc_null_band = np.percentile(acc_null, [2.5, 97.5], axis=0)  # The 95% confidence interval of the shuffles

        if excess:
            acc_mean = acc_mean - acc_null_mean
            acc_band_low = acc_mean - acc_sem
            acc_band_high = acc_mean + acc_sem
            ylabel = 'Excess accuracy'
            plt.axhline(0, color='tab:gray', linestyle='--')
            plt.plot(bin_centers, acc_mean, color='k', label='Excess acc.')
            plt.fill_between(bin_centers, acc_band_low, acc_band_high, color='tab:gray',
                             edgecolor='none', alpha=0.25, label='Acc. s.e.m.')
            plt.ylim(None, 0.5)
        else:
            # plt.plot(-np.mean(abs(pred_err), axis=0)+1)  # Equivalent
            plt.plot(bin_centers, acc_mean, color='k', label='Acc.')
            plt.fill_between(bin_centers, acc_mean - acc_sem, acc_mean + acc_sem, color='tab:gray', edgecolor='none', alpha=0.25,
                             label='Acc. s.e.m.')
            plt.plot(bin_centers, acc_null_mean, color='tab:gray', linestyle='--', label='Null mean')  # Chance level (0.5)
            plt.fill_between(bin_centers, acc_null_band[0], acc_null_band[1], color='tab:gray', edgecolor='none', alpha=0.25,
                             label='Null 95% CI')
            ylabel = 'Accuracy'
            plt.ylim(None, 1)

    plt.xlim(bins[0], bins[-1])
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    # plt.title(f'Decoding accuracy\n'
    #           f'{df_behavior.Subject.unique()[0]}, {acc.shape[0]} trials')
    # plt.legend(frameon=False)
    sns.despine()


def plot_mean_within_decoder(results, errorbar='ci', z_null=False, excess=True):
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
        plt.plot(bin_centers, z_scores_mean, color='k', label='Z acc.')
        p_values = p_val(acc_mean, acc_null_mean)
        significant_region = p_values < 0.05  # When assessing significance across sessions use p < 0.05
        # significant_region = np.abs(z_scores_mean) >= 1.96  # When assessing significance across sessions use 1.96
        plt.fill_between(bin_centers, z_scores_mean, where=significant_region, edgecolor='none', color='k',
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
        elif errorbar == 'sem':
            acc_band = (acc_mean - acc_sem, acc_mean + acc_sem)
            acc_null_band = (acc_null_mean - acc_null_sem, acc_null_mean + acc_null_sem)

        if excess:
            plt.axhline(0, color='tab:gray', linestyle='--')  # Chance level
            acc_mean = acc_mean - acc_null_mean
            acc_band = (acc_band[0] - acc_null_mean, acc_band[1] - acc_null_mean)
            yticks = (0, 0.25, 0.5)
            yticklabels = ('0', '0.25', '0.5')
            ylim = (-0.05, 0.5)
            ylabel = 'Excess accuracy'
        else:
            # Plot the mean null accuracy across all sessions (chance level)
            plt.plot(bin_centers, acc_null_mean, ls='--', color='tab:gray', label='Acc. null')
            plt.fill_between(bin_centers, acc_null_band[0], acc_null_band[1], color='tab:gray',
                             edgecolor='none', alpha=0.25)
            yticks = (0.5, 0.75, 1)
            yticklabels = ('0.5', '0.75', '1')
            ylim = (0.45, 1)
            ylabel = 'Accuracy'

        # Plot the mean decoding accuracy across all sessions
        plt.plot(bin_centers, acc_mean, color='k', label='Acc.')
        plt.fill_between(bin_centers, acc_band[0], acc_band[1], color='k', edgecolor='none',
                         alpha=0.25)

        # Plot the individual sessions accuracy
        for _ in range(len(results['acc'])):
            session_mean = np.mean(results['acc'][_], axis=0)
            if excess:
                session_mean = session_mean - acc_null_mean
            plt.plot(bin_centers, session_mean, color='tab:gray', alpha=0.1)

        # # Plot the individual sessions null accuracy (chance level)
        # for _ in range(len(results['acc_null'])):
        #     plt.plot(bin_centers, np.mean(results['acc_null'][_], axis=0), ls='--', color='tab:gray')

        # ylabel = 'Accuracy'

    plt.xlim(bins[0], bins[-1])
    plt.yticks(yticks, yticklabels)
    plt.ylim(ylim)
    plt.xlabel('Time (s) from stimulus')
    plt.ylabel(ylabel)
    # plt.title(f"Decoding accuracy\n"
    #           f"{subject}, {len(results['acc'])} sessions, {n_trials} trials")
    sns.despine()
    # plt.legend(frameon=False)

    return bin_centers, acc_mean, acc_band


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
    extent = [bins[0], bins[-1], bins[0], bins[-1]]
    x_min, x_max = extent[0], extent[1]

    if z_null:
        z_scores = [null_zscore(results['acc'][i], results['acc_null'][i]) for i in range(len(results['acc']))]
        z_scores_mean = np.mean((z_scores), axis=0)
        norm = CenteredNorm(vcenter=0, halfrange=4)
        plt.imshow(z_scores_mean, origin='lower', cmap='RdBu_r', norm=norm, extent=extent)
    else:
        plt.imshow(acc_mean, origin='lower', extent=extent)  # abs needed?

    plt.colorbar(label='Z-score')

    color = 'tab:gray'
    plt.axhline(0, color=color, linestyle='-')  # Stimulus / First lick
    plt.axvline(0, color=color, linestyle='-')  # Stimulus / First lick
    plt.axhline(1, color=color, linestyle='-')  # Go cue / ITI
    plt.axvline(1, color=color, linestyle='-')  # Go cue / ITI

    if align =='stim':
        plt.axhline(0.5, color=color, linestyle='--')  # Delay
        plt.axvline(0.5, color=color, linestyle='--')  # Delay
        epochs = {
            # 'stim': {'range': (0, 0.1), 'color': 'tab:blue', 'label': 'Stimulus'},
            # 'delay': {'range': (0.9, 1), 'color': 'tab:orange', 'label': 'Delay'},
            'stim': {'range': (0, 0.1), 'color': 'tab:blue', 'label': 'S'},
            'delay': {'range': (0.9, 1), 'color': 'tab:orange', 'label': 'D'},
            # 'resp': {'range': (1.85, 1.95), 'color': 'tab:green', 'label': 'Response'}
        }
        xlabel = 'Time (s) from stimulus'
        ylabel = 'Time (s) from stimulus'
    elif align == 'resp':
        epochs = {
            # 'first_lick': {'range': (-0.05, 0), 'color': 'darkgreen', 'label': 'First lick'},
            'first_lick': {'range': (-0.05, 0), 'color': 'darkgreen', 'label': 'First'},
            # 'mid_lick': {'range': (0.5, 0.55), 'color': 'lightgreen', 'label': 'Mid lick'},
            # 'last_lick': {'range': (0.95, 1), 'color': 'green', 'label': 'Last lick'},
            # 'post_lick': {'range': (1.5, 1.55), 'color': 'lightgreen', 'label': 'Post lick'}
            'post_lick': {'range': (1.5, 1.55), 'color': 'lightgreen', 'label': 'Post'}
        }
        xlabel = 'Time (s) from response'
        ylabel = 'Time (s) from response'

    ax = plt.gca()
    for name, props in epochs.items():
        start, end = props['range']
        height = end - start
        color = props['color']
        label = props['label']
        rect = patches.Rectangle(
            xy=(x_min, start),
            width=x_max - x_min,
            height=height,
            edgecolor=color,
            facecolor='none',
            zorder=2
        )
        ax.add_patch(rect)
        ax.text(
            x=x_min + bin_width,
            y=start + bin_width*2,
            s=label,
            color=color,
            ha='left',
            va='bottom'
        )

    first_tick = np.ceil(bins[0])  # Round up to the nearest integer
    last_tick = np.floor(bins[-1])  # Round down to the nearest integer
    ticks = np.arange(first_tick, last_tick + 1, 1)  # Create ticks at every integer value
    plt.xticks(ticks, ticks.astype(int))
    plt.yticks(ticks, ticks.astype(int))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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


def plot_mean_epoch_cross_decoder(results, epoch=None, engagement=None, excess=False, errorbar='ci', z_null=True):
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
    elif epoch == 'late_lick':
        color = 'lightgreen'
        label = 'Late lick'
    elif epoch == 'post_lick':
        color = 'lightgreen'
        label = 'Post lick'

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

    # plt.axvline(0, color='tab:gray', linestyle='-')  # Stimulus onset
    # plt.axvline(0.5, color='tab:gray', linestyle='--')  # Delay
    # plt.axvline(1, color='tab:gray', linestyle='-')  # Go cue

    if excess:
        acc_mean = acc_mean - acc_null_mean
        acc_band = (acc_band[0] - acc_null_mean, acc_band[1] - acc_null_mean)
        yticks = (0, 0.25, 0.5)
        yticklabels = ('0', '0.25', '0.5')
        ylim = (-0.05, 0.5)
        ylabel = 'Excess accuracy'

    else:
        plt.plot(bin_centers, acc_null_mean, linestyle='--', color=color)
        plt.fill_between(bin_centers, acc_null_band[0], acc_null_band[1], color=color, edgecolor='none',
                         alpha=0.25)
        yticks = (0.5, 0.75, 1)
        yticklabels = ('0.5', '0.75', '1')
        ylim = (0.45, 1)
        ylabel = 'Accuracy'

    # n_trials = np.sum([results['pred'][i].shape[0] for i in range(len(results['acc']))])
    # plt.figure(constrained_layout=True)
    plt.plot(bin_centers, acc_mean, color=color, label=label)
    plt.fill_between(bin_centers, acc_band[0], acc_band[1], color=color, edgecolor='none', alpha=0.25)
    plt.xlim(bins[0], bins[-1])
    plt.xticks(np.arange(bins[0], bins[-1] + 1, 1))
    xlabel = 'Time (s) from stimulus' if epoch in ['stim', 'delay'] else 'Time (s) from response'
    plt.xlabel(xlabel)
    plt.yticks(yticks, yticklabels)
    plt.ylim(ylim)
    plt.ylabel(ylabel)

    # plt.title(f'Decoding accuracy\n'
    #           f'{subject}, {len(results["acc"])} sessions, {n_trials} trials')
    plt.legend(frameon=False)
    sns.despine()


def plot_mean_epoch_cross_decoder_split(results, what='stim', epoch=None, split='hit', excess=True, errorbar='ci', z_null=True):
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
    acc_null1_sem = sem([np.nanmean(results['acc_null'][i][1], axis=(0, 2)) for i in range(len(results['acc_null']))],
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

    if split == 'hit':
        colors = ('tab:red', 'tab:green')
        labels = ('Error', 'Correct')
        # Flip errors to make it easier to compare (only when trained in correct trials only)
        if ((what == 'stim' and epoch in ['delay', 'resp']) or
            (what == 'choice' and epoch == 'stim')):
            # chance = acc_null0_mean
            chance = 0.5
            acc0_mean = chance + (chance - acc0_mean)
            acc_null0_mean = chance + (chance - acc_null0_mean)
            acc0_band = (chance + (chance - acc0_band[1]), chance + (chance - acc0_band[0]))
            acc_null0_band = (chance + (chance - acc_null0_band[1]), chance + (chance - acc_null0_band[0]))

    elif split == 'engagement':
        # Align to stimulus
        if epoch == 'stim':
            colors = ('tab:gray', 'tab:blue')
        elif epoch == 'delay':
            colors = ('tab:gray', 'tab:orange')
        elif epoch == 'resp':
            colors = ('tab:gray', 'tab:green')
        # Align to first lick
        elif epoch == 'first_lick':
            colors = ('tab:gray', 'darkgreen')
        elif epoch == 'mid_lick':
            colors = ('tab:gray', 'lightgreen')
        elif epoch == 'late_lick':
            colors = ('tab:gray', 'tab:green')
        elif epoch == 'post_lick':
            colors = ('tab:gray', 'tab:green')
        labels = ('Disengaged', 'Engaged')

    # plt.figure(constrained_layout=True)
    plt.axvline(0, color='tab:gray', linestyle='-')  # Stimulus / First lick
    if epoch in ['stim', 'delay', 'resp']:
        plt.axvline(0.5, color='tab:gray', linestyle='--')  # Delay
    plt.axvline(1, color='tab:gray', linestyle='-')  # Go cue / ITI

    bins = results['bins']
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if excess:
        plt.axhline(0, color='tab:gray', linestyle='--')  # Chance level
        acc0_mean = acc0_mean - acc_null0_mean
        acc1_mean = acc1_mean - acc_null1_mean
        acc0_band = (acc0_band[0] - acc_null0_mean, acc0_band[1] - acc_null0_mean)
        acc1_band = (acc1_band[0] - acc_null1_mean, acc1_band[1] - acc_null1_mean)
        # ylim = (0, 0.5)
        ylim = (0, None)
        ylabel = 'Excess accuracy'
    else:
        plt.plot(bin_centers, acc_null0_mean, color=colors[0], linestyle='--')
        # plt.fill_between(bin_centers, acc_null0_band[0], acc_null0_band[1], color=colors[0], edgecolor='none',
        #                  alpha=0.25)
        plt.plot(bin_centers, acc_null1_mean, color=colors[1], linestyle='--')
        # plt.fill_between(bin_centers, acc_null1_band[0], acc_null1_band[1], color='tab:green', edgecolor='none',
        #                     alpha=0.25)
        ylim = (0.5, 1)
        ylabel = 'Accuracy'

    # Condition 0 (error/disengaged)
    plt.plot(bin_centers, acc0_mean, color=colors[0], label=labels[0])
    plt.fill_between(bin_centers, acc0_band[0], acc0_band[1], color=colors[0],
                     edgecolor='none', alpha=0.25)

    # Condition 1 (correct/engaged). Negative to flip it and make it easier to compare
    plt.plot(bin_centers, acc1_mean, color=colors[1], label=labels[1])
    plt.fill_between(bin_centers, acc1_band[0], acc1_band[1], color=colors[1],
                     edgecolor='none', alpha=0.25)

    plt.xlim(bins[0], bins[-1])
    plt.xticks(np.arange(bins[0], bins[-1] + 1, 1))
    # plt.ylim(None, 1)
    xlabel = 'Time (s) from stimulus' if epoch in ['stim', 'delay'] else 'Time (s) from response'
    plt.xlabel(xlabel)
    # plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    sns.despine()


def plot_weights_dist(cluster_info, ipsi_contra=False):

    weights = cluster_info['weights'].to_numpy()
    selective_mask = cluster_info['selective_mask'].to_numpy()
    rec_side = cluster_info['RECside'].to_numpy()  # 1 for right, 0 for left

    # Flip weights for ipsi/contra convention if requested
    if ipsi_contra:
        weights = lr2ic(weights, rec_side)
        labels = ('Contra', 'Ipsi')
    else:
        labels = ('Left', 'Right')

    non_selective = ~selective_mask
    negative_selective = selective_mask & (weights < 0)  # Left/Ipsi
    positive_selective = selective_mask & (weights > 0)  # Right/Contra

    # Plot single histogram, color by selectivity
    plt.figure(figsize=fig_size(n_cols=2), constrained_layout=True)
    bins = np.linspace(-1, 1, 50)

    # Plot
    plt.hist(weights[non_selective], bins=bins, weights=np.ones(non_selective.sum())/len(weights),
             color='lightgray', edgecolor='k')
    plt.hist(weights[negative_selective], bins=bins, weights=np.ones(negative_selective.sum()) / len(weights),
             color='tab:blue', edgecolor='k', label=labels[0], alpha=0.75)
    plt.hist(weights[positive_selective], bins=bins, weights=np.ones(positive_selective.sum()) / len(weights),
             color='tab:orange', edgecolor='k', label=labels[1], alpha=0.75)

    plt.axvline(0, color='k', linestyle='--')
    plt.xlim(-1, 1)
    xticks = np.arange(-1, 1.01, 0.5)
    xticklabels = [f'{x:g}' for x in xticks]  # 'g' removes trailing zeros
    plt.xticks(xticks, xticklabels)
    plt.title(f'{selective_mask.sum() / len(weights) * 100:.0f}%')
    plt.xlabel('Weight')
    plt.ylabel('Density')
    # plt.legend(loc='upper center', ncol=2)
    ylim = plt.gca().get_ylim()[1] * 0.75
    plt.text(-0.5, ylim, labels[0], color='tab:blue', ha='center', va='center')
    plt.text(0.5, ylim, labels[1], color='tab:orange', ha='center', va='center')
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


# stim_len = df.StimLen.unique()[0]
# delay = df.Delay.unique()[0]
# resp_win = df.RespWinLen.unique()[0]
# timeout = df.Timeout.unique()[0]
# iti = df.ITI.unique()[0]
# load_time = 2.5
#
# total = stim_len + delay + resp_win + timeout + iti + load_time
# print(f'Trial total time: {total}s')


# subjects = ['000', '007', '009']  # Removed 001 (all sessions bad)
# folder_parent = Path.home() / 'data'
# # summaries = pd.DataFrame()
# # df = pd.DataFrame()
# for subj in subjects:
    # df = pd.concat([df, get_all_beh(subj)], ignore_index=True)
    # summaries = pd.concat([summaries, get_beh(subj)], ignore_index=True)
    #
    # # Preprocess
    # preprocess_subject(subj)
    #
    # Within decoder
    # mean_decoder(subj, what='stim', align='stim', kind='within', epoch=None, epoch_ortho=None, split_by=None,
    #              drop_miss=True, hit_only=True, engagement=1, n_shuffles=100, save=True)
    #
    # X decoder
    # mean_decoder(subj, what='stim', align='stim', kind='cross', epoch=None, epoch_ortho=None, split_by=None, drop_miss=True,
    #              hit_only=True, engagement=1, n_shuffles=100, save=True)
    # mean_decoder(subj, what='stim', align='resp', kind='cross', epoch=None, epoch_ortho=None, split_by=None, drop_miss=True,
    #              hit_only=True, engagement=1, n_shuffles=100, save=True)
    #
    # Epoch decoders (align to stimulus onset)
    # mean_decoder(subj, what='stim', align='stim', kind='epoch', epoch='stim', epoch_ortho=None, split_by=None,
    #              drop_miss=True, hit_only=True, engagement=1, n_shuffles=100, save=True)
    # mean_decoder(subj, what='stim', align='stim', kind='epoch', epoch='delay', epoch_ortho=None, split_by=None,
    #              drop_miss=True, hit_only=True, engagement=1, n_shuffles=100, save=True)
    # mean_decoder(subj, what='stim', align='stim', kind='epoch', epoch='resp', epoch_ortho=None, split_by=None,
    #              drop_miss=True, hit_only=True, engagement=1, n_shuffles=100, save=True)

    # Epoch decoders (align to first lick)
    # mean_decoder(subj, what='choice', align='resp', kind='epoch', epoch='first_lick', epoch_ortho=None, split_by=None,
    #              drop_miss=True, hit_only=True, engagement=1, n_shuffles=100, save=True)
    # mean_decoder(subj, what='choice', align='resp', kind='epoch', epoch='mid_lick', epoch_ortho=None, split_by=None,
    #              drop_miss=True, hit_only=True, engagement=1, n_shuffles=100, save=True)

    # # Split by outcome (generalize)
    # mean_decoder(subj, what='stim', align='stim', kind='epoch_generalize', epoch='stim', epoch_ortho=None, split_by='Hit', drop_miss=True,
    #                  hit_only=False, engagement=1, n_shuffles=100, save=True)
    # mean_decoder(subj, what='choice', align='stim', kind='epoch_generalize', epoch='delay', epoch_ortho=None, split_by='Hit', drop_miss=True,
    #                  hit_only=False, engagement=1, n_shuffles=100, save=True)
    # mean_decoder(subj, what='choice', align='stim', kind='epoch_generalize', epoch='resp', epoch_ortho=None, split_by='Hit', drop_miss=True,
    #                  hit_only=False, engagement=1, n_shuffles=100, save=True)

    # mean_decoder(subj, what='choice', align='resp', kind='epoch_generalize', epoch='first_lick', epoch_ortho=None, split_by='Hit',
    #              drop_miss=True, hit_only=False, engagement=1, n_shuffles=100, save=True)
    # mean_decoder(subj, what='choice', align='resp', kind='epoch_generalize', epoch='mid_lick', epoch_ortho=None, split_by='Hit',
    #              drop_miss=True, hit_only=False, engagement=1, n_shuffles=100, save=True)

    # # Split by engagement
    # mean_decoder(subj, what='stim', kind='epoch_split', epoch='stim', epoch_ortho=None, split_by='Engaged', drop_miss=True,
    #                  hit_only=True, engagement=None, n_shuffles=100, plot=False, save=True)
    # mean_decoder(subj, what='choice', kind='epoch_split', epoch='delay', epoch_ortho=None, split_by='Engaged', drop_miss=True,
    #                  hit_only=True, engagement=None, n_shuffles=100, plot=False, save=True)
    # mean_decoder(subj, what='choice', kind='epoch_split', epoch='resp', epoch_ortho=None, split_by='Engaged', drop_miss=True,
    #                  hit_only=True, engagement=None, n_shuffles=100, plot=False, save=True)
    #
    # # Split by engagement (generalize)
    # mean_decoder(subj, what='stim', align='stim', kind='epoch_generalize', epoch='stim', epoch_ortho=None, split_by='Engaged',
    #              drop_miss=True, hit_only=True, engagement=None, n_shuffles=100, plot=False, save=True)
    # mean_decoder(subj, what='choice', align='stim', kind='epoch_generalize', epoch='delay', epoch_ortho=None, split_by='Engaged',
    #              drop_miss=True, hit_only=True, engagement=None, n_shuffles=100, plot=False, save=True)
    # mean_decoder(subj, what='choice', align='stim', kind='epoch_generalize', epoch='resp', epoch_ortho=None, split_by='Engaged',
    #              drop_miss=True, hit_only=True, engagement=None, n_shuffles=100, plot=False, save=True)

    # mean_decoder(subj, what='choice', align='resp', kind='epoch_generalize', epoch='first_lick', epoch_ortho=None, split_by='Engaged',
    #              drop_miss=True, hit_only=True, engagement=None, n_shuffles=100, plot=False, save=True)
    # mean_decoder(subj, what='choice', align='resp', kind='epoch_generalize', epoch='mid_lick', epoch_ortho=None, split_by='Engaged',
    #              drop_miss=True, hit_only=True, engagement=None, n_shuffles=100, plot=False, save=True)


# Paralelization test
# subjects = ['000', '007', '009']
# folder_parent = Path.home() / 'data'
# def run_fun(subj):
#     # mean_decoder(subj, what='stim', align='resp', kind='cross', epoch=None, epoch_ortho=None,
#     #              split_by=None, drop_miss=True, hit_only=True, engagement=1, n_shuffles=100,
#     #              plot=False, save=True)
#     preprocess_subject(subj, align='stim', time_win=[-1, 3], bin_size=0.1)
# Parallel(n_jobs=-1)(delayed(run_fun)(subj) for subj in subjects)