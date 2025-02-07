from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import sem
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





bins, all_psth = get_all_psth(df_spikes, cluster_info, n_trials, group='good', time_win=[-1, 3], bin_size=0.1)


def decode_condition(X=np.zeros((1, 1)), y=np.zeros((1, 1)), n_shuffles=100):
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
    null_acc = np.empty((n_shuffles, X.shape[1]))

    # Apply z-scoring normalization across neurons and time bins (per trial)
    scaler = StandardScaler()

    # Cross-validate results
    # kf = StratifiedKFold(n_splits=5, shuffle=True) # Stratified cross-validation
    kf = KFold()  # K-Fold cross-validation

    # Split trials into training and testing sets (each fold gets a unique test set to prevent overfitting)
    for train_index, test_index in kf.split(X):

        # Loop over each time bin
        for bin in range(X.shape[1]):

            # Define train and testing set for the current time bin and fold
            X_train, X_test = X[train_index, bin], X[test_index, bin]
            y_train, y_test = y[train_index], y[test_index]

            # Apply z-scoring normalization to the current time bin's data (otherwise might not converge)
            X_train = scaler.fit_transform(X_train)  # Fit and transform on the training set
            X_test = scaler.transform(X_test)  # Only transform the test set using the same scaler

            # Train decoder (logistic regression) on the current time bin’s neural activity
            clf = LogisticRegression()
            clf.fit(X_train, y_train)

            # Evaluate decoder
            y_pred = clf.predict(X_test)  # Predicts the stimulus category for test trials
            y_acc = accuracy_score(y_test, y_pred)  # Computes accuracy for each fold & time bin
            # print(f"Accuracy: {y_acc:.2f}")

            # Store results
            pred[test_index, bin] = y_pred  # Predicted stimulus condition for each test trial at each time bin
            pred_err[test_index, bin] = y_pred - y_test  # Difference between predicted and actual labels
            acc[test_index, bin] = y_acc  # Accuracy for each test trial at each time bin

            # Compute null distribution by shuffling the labels and evaluating accuracy
            y_test_shuffled = y_test.values.copy()
            for _ in range(n_shuffles):
                np.random.shuffle(y_test_shuffled)
                null_acc[_, bin] = accuracy_score(y_test_shuffled, y_pred)

    return pred, pred_err, acc, null_acc


pred, pred_err, acc, null_acc = decode_condition(X=all_psth, y=df_behavior.Side, n_shuffles=100)


def plot_decoding_acc(acc, null_acc):

    acc_mean = acc.mean(axis=0)
    acc_sem = sem(acc, axis=0)
    acc_null_mean = null_acc.mean(axis=0)
    acc_null_band = np.percentile(null_acc, [2.5, 97.5], axis=0)  # The 95% confidence interval of the shuffles

    # Plot decoding accuracy
    plt.figure(constrained_layout=True)
    # plt.plot(-np.mean(abs(pred_err), axis=0)+1)  # Equivalent
    plt.plot(bins[:-1], acc_mean)
    plt.fill_between(bins[:-1], acc_mean - acc_sem, acc_mean + acc_sem, alpha=0.25)
    plt.plot(bins[:-1], acc_null_mean)
    plt.fill_between(bins[:-1], acc_null_band[0], acc_null_band[1], alpha=0.25)

    plt.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
    plt.axvline(0, color='tab:red', linestyle='-')  # Stimulus onset
    plt.axvline(go_cue, color='tab:gray', linestyle='-')  # Go cue
    # Set upper ylim to 1 for better visualization but keep the lower ylim as it is
    # plt.ylim(plt.gca().get_ylim()[0], 1)
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.title('Decoding accuracy')
    sns.despine()


pred = []
pred_err = []
acc = []
null_acc = []
n_trials = 0

for i in range(len(ephys_ids)):

    id = ephys_ids[i]
    path_behavior = (Path.home() / 'Downloads' / behavior_ids[i]).with_suffix('.csv')
    df_ttl, df_behavior, n_trials, df_spikes, cluster_info, x, height, labels, y, width, left, ts_edges, events_edges = \
        preprocess(id, path_behavior)
    bins, all_psth = get_all_psth(df_spikes, cluster_info, n_trials, group='good', time_win=[-1, 3], bin_size=0.1)
    pred_, pred_err_, acc_, null_acc_ = decode_condition(X=all_psth, y=df_behavior.Side, n_shuffles=100)
    pred.append(pred_)
    pred_err.append(pred_err_)
    acc.append(acc_)
    null_acc.append(null_acc_)
    n_trials += n_trials

for i in range(len(ephys_ids)):
    plot_decoding_acc(acc[i], null_acc[i])


# Compute the mean accuracy across all sessions using list comprehension
acc_mean = [acc[i].mean(axis=0) for i in range(len(acc))]
acc_mean = np.array(acc_mean)
acc_null_mean = [null_acc[i].mean(axis=0) for i in range(len(null_acc))]
acc_null_mean = np.array(acc_null_mean)
n_trials = np.sum([acc[i].shape[0] for i in range(len(acc))])


plt.figure(constrained_layout=True)

# Plot the mean decoding accuracy across all sessions
plt.plot(bins[:-1], np.mean(acc_mean, axis=0))
plt.fill_between(bins[:-1], np.mean(acc_mean, axis=0) - sem(acc_mean, axis=0),
                 np.mean(acc_mean, axis=0) + sem(acc_mean, axis=0), alpha=0.25)

# Plot the mean null accuracy across all sessions (chance level)
plt.plot(bins[:-1], np.mean(acc_null_mean, axis=0), ls='--', c='tab:gray')
plt.fill_between(bins[:-1], np.percentile(acc_null_mean, 2.5, axis=0), np.percentile(acc_null_mean, 97.5, axis=0),
                 color='tab:gray', alpha=0.25)

# plt.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
plt.axvline(0, color='tab:red', linestyle='-')  # Stimulus onset
plt.axvline(go_cue, color='tab:gray', linestyle='-')  # Go cue
# Set upper ylim to 1 for better visualization but keep the lower ylim as it is
# plt.ylim(plt.gca().get_ylim()[0], 1)
plt.xlabel('Time (s)')
plt.ylabel('Accuracy')
plt.title(f'Decoding accuracy (mouse {df_behavior.Subject.unique()[0]}, {n_trials} trials)')
sns.despine()