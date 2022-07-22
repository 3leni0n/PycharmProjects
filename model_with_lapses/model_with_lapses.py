import numpy as np
from scipy import stats
from scipy.optimize import minimize
from my_fun.my_fun import *  # Or from daily_report.daily_report import daily_report


x = np.array(stim_strength)
# x = x[:, 1:4]
y = np.array(choices)



def fit_model(x, y, iterations=5):
    """Computes a psychometric function.
    x is a vector
    """
    # https://psychology.stackexchange.com/questions/13347/how-can-i-fit-a-psychometric-function-such-that-the-minimum-is-50-chance-level

    frame1 = x[:, 0]
    frame2 = x[:, 1]
    frame3 = x[:, 2]
    frame4 = x[:, 3]
    frame5 = x[:, 4]
    frame6 = x[:, 5]
    frame7 = x[:, 6]
    frame8 = x[:, 7]
    frame9 = x[:, 8]
    frame10 = x[:, 9]

    def sigmoid_mme(fit_params: tuple):
        lapse1, lapse2, bias, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = fit_params

        # Function to fit:
        y_pred = lapse1 + (1 - lapse1 - lapse2) / (1 + np.exp(-(bias + k1*frame1 + k2*frame2 + k3*frame3 + k4*frame4 + k5*frame5 +
                                                 k6*frame6 + k7*frame7 + k8*frame8 + k9*frame9 + k10*frame10)))

        # Calculate negative log likelihood:
        ll = - np.sum(stats.norm.logpdf(y, loc=y_pred))

        return ll

    # coherence_dataframe = pd.DataFrame({'r_resp': y, 'evidence': x})
    #
    # info = coherence_dataframe.groupby(['evidence'])['r_resp'].mean()
    # ydata = [np.around(elem, 3) for elem in info.values]
    # xdata = info.index.values
    # fit_error = [np.around(elem, 3) for elem in coherence_dataframe.groupby(['evidence'])['r_resp'].sem().values]

    best_ll = 1000000000
    best_fit = None

    for i in range(iterations):
        initial_guess_lapses = np.random.random(2)
        initial_guess_other_params = (np.random.random(11) * 2 - 1)  # So that is between -1 and 1
        initial_guess = np.hstack([initial_guess_lapses, initial_guess_other_params])
        # Run the minimizer:
        ll = minimize(sigmoid_mme, initial_guess)

        print(ll.fun)

        if ll.fun < best_ll:
            best_fit = ll
            best_ll = ll.fun

        print(best_ll)
        print("")
    return best_fit

def fit_model_without_lapses(x, y, iterations=5):
    """Computes a psychometric function.
    x is a vector
    """
    # https://psychology.stackexchange.com/questions/13347/how-can-i-fit-a-psychometric-function-such-that-the-minimum-is-50-chance-level

    frame1 = x[:, 0]
    frame2 = x[:, 1]
    frame3 = x[:, 2]
    frame4 = x[:, 3]
    frame5 = x[:, 4]
    frame6 = x[:, 5]
    frame7 = x[:, 6]
    frame8 = x[:, 7]
    frame9 = x[:, 8]
    frame10 = x[:, 9]

    def sigmoid_mme(fit_params: tuple):
        bias, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = fit_params

        # Function to fit:
        y_pred = 1 / (1 + np.exp(-(bias + k1*frame1 + k2*frame2 + k3*frame3 + k4*frame4 + k5*frame5 +
                                                 k6*frame6 + k7*frame7 + k8*frame8 + k9*frame9 + k10*frame10)))

        # Calculate negative log likelihood:
        ll = - np.sum(stats.norm.logpdf(y, loc=y_pred))

        return ll

    # coherence_dataframe = pd.DataFrame({'r_resp': y, 'evidence': x})
    #
    # info = coherence_dataframe.groupby(['evidence'])['r_resp'].mean()
    # ydata = [np.around(elem, 3) for elem in info.values]
    # xdata = info.index.values
    # fit_error = [np.around(elem, 3) for elem in coherence_dataframe.groupby(['evidence'])['r_resp'].sem().values]

    best_ll = 1000000000
    best_fit = None

    for i in range(iterations):
        initial_guess = np.random.random(11) * 2 - 1  # So that is between -1 and 1
        # Run the minimizer:
        ll = minimize(sigmoid_mme, initial_guess)

        print(ll.fun)

        if ll.fun < best_ll:
            best_fit = ll
            best_ll = ll.fun

        print(best_ll)
        print("")
    return best_fit

def fit_model_simple(x, y, iterations=5):
    """Computes a psychometric function.
    x is a vector
    """
    # https://psychology.stackexchange.com/questions/13347/how-can-i-fit-a-psychometric-function-such-that-the-minimum-is-50-chance-level

    frame1 = x[:, 0]
    frame2 = x[:, 1]
    frame3 = x[:, 2]


    def sigmoid_mme(fit_params: tuple):
        lapse1, lapse2, bias, k1, k2, k3 = fit_params

        # Function to fit:
        y_pred = lapse1 + (1 - lapse1 - lapse2) / (1 + np.exp(-(bias + k1*frame1 + k2*frame2 + k3*frame3)))

        # Calculate negative log likelihood:
        ll = - np.sum(stats.norm.logpdf(y, loc=y_pred))

        return ll

    # coherence_dataframe = pd.DataFrame({'r_resp': y, 'evidence': x})
    #
    # info = coherence_dataframe.groupby(['evidence'])['r_resp'].mean()
    # ydata = [np.around(elem, 3) for elem in info.values]
    # xdata = info.index.values
    # fit_error = [np.around(elem, 3) for elem in coherence_dataframe.groupby(['evidence'])['r_resp'].sem().values]

    best_fit_value = 1000000000
    best_fit = None

    for i in range(iterations):
        initial_guess = np.random.random(6) * 2 - 1
        # Run the minimizer:
        ll = minimize(sigmoid_mme, initial_guess)

        print(ll.fun)

        if ll.fun < best_fit_value:
            best_fit = ll
            best_fit_value = ll.fun

        print(best_fit_value)
        print("")
    return best_fit
