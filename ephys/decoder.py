from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import sem
from my_fun.my_fun import timer
from ephys.preprocessing import *
from ephys.analysis import *


# Neuromatch tutorial https://compneuro.neuromatch.io/tutorials/W1D5_DeepLearning/student/W1D5_Tutorial1.html

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


# Get behavioral events
stim_dur = df_behavior.StimDur.unique()[0]
delay = df_behavior.Delay.unique()[0]
go_cue = stim_dur + delay


# Create ndimensional array with all PSTHs (trials x time x neurons)
bins, all_psth = get_all_psth(cluster_info, df_spikes, df_ttl, n_trials, time_win=[-1, 3], bin_size=0.1)

# good_clusters = select_cluster_index(cluster_info, group='good')


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
    pred = np.empty((X.shape[0], X.shape[1], X.shape[1]) )
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
                pred[test_index, bin_train, bin_test] = y_pred  # Predicted stimulus condition for each test trial at each time bin_train
                pred_err[test_index, bin_train, bin_test] = y_pred - y_test  # Difference between predicted and actual labels
                acc[test_index, bin_train, bin_test] = y_acc  # Accuracy for each test trial at each time bin_train

                # Compute null distribution by shuffling the labels and evaluating accuracy
                y_test_shuffled = y_test.values.copy()
                for _ in range(n_shuffles):
                    np.random.shuffle(y_test_shuffled)
                    acc_null[_, bin_train, bin_test] = accuracy_score(y_test_shuffled, y_pred)

    return pred, pred_err, acc, acc_null


# pred, pred_err, acc, acc_null = within_decoder(X=all_psth, y=df_behavior.Side, n_shuffles=1000)
# pred, pred_err, acc, acc_null = cross_decoder(X=all_psth, y=df_behavior.Side, n_shuffles=10)


def plot_cross_decoder(pred_err):
    plt.figure(constrained_layout=True)
    plt.imshow(np.mean(abs(pred_err), axis=0), origin='lower')
    plt.colorbar()
    plt.xticks(np.arange(0, len(bins), 10), np.round(bins[::10]))
    plt.yticks(np.arange(0, len(bins), 10), np.round(bins[::10]))
    plt.axhline(np.where(bins == 0)[0], color='k', linestyle='-')  # Stimulus onset
    plt.axvline(np.where(bins == 0)[0], color='k', linestyle='-')  # Stimulus onset
    plt.axhline(np.where(bins == go_cue)[0], color='k', linestyle='-')  # Go cue
    plt.axvline(np.where(bins == go_cue)[0], color='k', linestyle='-')  # Go cue
    plt.xlabel('Test time (s)')
    plt.ylabel('Train time (s)')
    plt.title('Cross temporal decoder')
    sns.despine()


def null_zscore(acc, acc_null):
    """
    Compute and z-scores and p-values for the decoding accuracy relative to the null distribution.
    :param acc: 2D array with decoding accuracy (trials x time)
    :param acc_null: 3D array with null distribution of accuracy (shuffles x time)
    :return: z_scores, p_values
    """

    # Compute p-values and z-scores
    acc_mean = np.mean(acc, axis=0)  # Mean accuracy across trials
    null_acc_mean = np.mean(acc_null, axis=0)  # Mean accuracy across shuffles
    null_acc_std = np.std(acc_null, axis=0)  # Standard deviation of accuracy across shuffles
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
    p_values = np.mean(acc_null > acc_mean, axis=0)  # p-value as the fraction of shuffles where null accuracy > real accuracy
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
        plt.fill_between(bins[:-1], acc_mean - acc_sem, acc_mean + acc_sem, edgecolor='none', alpha=0.25, label='Acc. s.e.m.')
        plt.plot(bins[:-1], acc_null_mean, color='tab:gray', linestyle='-', label='Null mean')  # Chance level (0.5)
        plt.fill_between(bins[:-1], acc_null_band[0], acc_null_band[1], color='tab:gray', edgecolor='none', alpha=0.25,
                         label='Null 95% CI')
        ylabel = 'Accuracy'

    plt.axvline(0, color='tab:red', linestyle='-')  # Stimulus onset
    plt.axvline(go_cue, color='tab:gray', linestyle='-')  # Go cue
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(f'Decoding accuracy\n'
              f'{df_behavior.Subject.unique()[0]}, session {i+1}, {acc.shape[0]} trials')
    plt.legend(frameon=False)
    sns.despine()


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
        'acc_null': []
    }

    for i in range(len(ephys_ids)):
        print(f'Processing session {i + 1}/{len(ephys_ids)}...')
        id = ephys_ids[i]
        path_behavior = (Path.home() / 'Downloads' / behavior_ids[i]).with_suffix('.csv')
        df_ttl, df_behavior, n_trials, df_spikes, cluster_info, x, height, labels, y, width, left, ts_edges, events_edges = \
            preprocess(id, path_behavior)
        bins, all_psth = get_all_psth(cluster_info, df_spikes, df_ttl, n_trials, time_win=[-1, 3], bin_size=0.1)
        pred, pred_err, acc, acc_null = within_decoder(X=all_psth, y=df_behavior.Side, n_shuffles=1000)

        results['pred'].append(pred)
        results['pred_err'].append(pred_err)
        results['acc'].append(acc)
        results['acc_null'].append(acc_null)

        # Plot decoding accuracy for each session
        if plot:
            plot_within_decoder(bins, results['acc'][i], results['acc_null'][i], z_null=True)

    return results


# results = mean_within_decoder(plot=False)


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

    if z_null:
        z_scores = [null_zscore(results['acc'][0], results['acc_null'][0]) for i in range(len(results['acc']))]
        z_scores_mean = np.mean(z_scores, axis=0)
        plt.plot(bins[:-1], z_scores_mean, label='Z acc.')
        # p_values = p_val(acc_mean, acc_null_mean)
        # significant_region = p_values < 0.05  # When assessing significance across sessions use p < 0.05
        significant_region = np.abs(z_scores_mean) >= 1.96  # When assessing significance across sessions use 1.96
        plt.fill_between(bins[:-1], z_scores_mean, where=significant_region, edgecolor='none',
                         alpha=0.25, label='α < .05')
        plt.axhline(0, color='tab:gray', linestyle='--')  # Chance level
        plt.axhline(1.96, color='tab:gray', linestyle='--')  # 95% confidence interval
        ylabel = 'Z-score'

    else:
        # Plot the mean decoding accuracy across all sessions
        plt.plot(bins[:-1], np.mean(acc_mean, axis=0))
        plt.fill_between(bins[:-1], np.mean(acc_mean, axis=0) - sem(acc_mean, axis=0),
                         np.mean(acc_mean, axis=0) + sem(acc_mean, axis=0), edgecolor='none', alpha=0.25)

        # Plot the mean null accuracy across all sessions (chance level)
        plt.plot(bins[:-1], np.mean(acc_null_mean, axis=0), ls='--', c='tab:gray')
        plt.fill_between(bins[:-1], np.percentile(acc_null_mean, 2.5, axis=0), np.percentile(acc_null_mean, 97.5, axis=0),
                         color='tab:gray', edgecolor='none', alpha=0.25)
        ylabel = 'Accuracy'

    plt.axvline(0, color='tab:red', linestyle='-')  # Stimulus onset
    plt.axvline(go_cue, color='tab:gray', linestyle='-')  # Go cue
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(f"Decoding accuracy\n"
              f"{df_behavior.Subject.unique()[0]}, {len(results['acc'])} sessions, {n_trials} trials")
    sns.despine()


# Select spikes from df_spikes that belong to good clusters from cluster_info
good_clusters = cluster_info[cluster_info.group == 'good'].cluster_id