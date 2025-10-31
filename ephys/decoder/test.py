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


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

def train_decoder(X_train, y_train):
    """
    Fit a logistic regression decoder on one time bin of training data.
    Returns the fitted scaler and classifier.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit scaler on training data
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf, scaler


def test_decoder(clf, scaler, X_test, y_test, n_shuffles=100):
    """
    Test a trained decoder on one time bin of testing data.
    Returns predictions, prediction errors, accuracy, and shuffled null accuracy.
    """
    # Transform test data using the same scaler as training
    X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)
    y_acc = accuracy_score(y_test, y_pred)

    # Compute null accuracy by shuffling labels
    acc_null = np.empty(n_shuffles)
    y_test_shuffled = y_test.values.copy() if hasattr(y_test, 'values') else y_test.copy()
    for i in range(n_shuffles):
        np.random.shuffle(y_test_shuffled)
        acc_null[i] = accuracy_score(y_test_shuffled, y_pred)

    pred_err = y_pred - y_test
    return y_pred, pred_err, y_acc, acc_null


def cross_decoder(X=np.zeros((1, 1, 1)), y=np.zeros((1, 1)), n_shuffles=100):
    """
    Perform cross-temporal decoding using logistic regression with stratified K-fold validation.
    Splits training and testing logic into modular functions.
    """

    n_trials, n_bins, _ = X.shape
    pred = np.empty((n_trials, n_bins, n_bins))
    pred_err = np.empty((n_trials, n_bins, n_bins))
    acc = np.empty((n_trials, n_bins, n_bins))
    acc_null = np.empty((n_shuffles, n_bins, n_bins))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(X, y):
        # Loop over all training bins
        for bin_train in range(n_bins):
            X_train, y_train = X[train_index, bin_train], y[train_index]
            clf, scaler = train_decoder(X_train, y_train)

            # Loop over all testing bins
            for bin_test in range(n_bins):
                X_test, y_test = X[test_index, bin_test], y[test_index]
                y_pred, y_pred_err, y_acc, acc_null_dist = test_decoder(
                    clf, scaler, X_test, y_test, n_shuffles
                )

                pred[test_index, bin_train, bin_test] = y_pred
                pred_err[test_index, bin_train, bin_test] = y_pred_err
                acc[test_index, bin_train, bin_test] = y_acc
                acc_null[:, bin_train, bin_test] = acc_null_dist

    return pred, pred_err, acc, acc_null
