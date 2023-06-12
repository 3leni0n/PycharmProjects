import pandas as pd
import numpy as np
import statsmodels.api as sm


# Define functions to make design matrices
def make_choice_history_dm(df, k):
    """
    Make a design matrix with the choice history. There is a column for each previous trial (up to k). In each trial,
    only one of these regressors is non-zero.
    :param df: DataFrame with hit and choice data
    :param k: Number of trials to look back
    :return: Design matrix
    """
    def get_choice_history(df, k):
        """
        Get the choice history for a trial number k.
        :param df: DataFrame with hit and choice data
        :param k: Number of trials to look back
        :return:  r_minus, r_plus
        """
        r_minus = []
        r_plus = []
        for _ in range(len(df)):
            if _ < k:
                r_minus.append(np.nan)
                r_plus.append(np.nan)
            else:
                # r-(t-k): error right = +1, error left = -1, no error (correct) = 0
                if df.Hit[_ - k] == 0 and df.Choice[_ - k] == 1:
                    r_minus.append(1)
                elif df.Hit[_ - k] == 0 and df.Choice[_ - k] == 0:
                    r_minus.append(-1)
                elif df.Hit[_ - k] == 1:
                    r_minus.append(0)
                # r+(t-k): correct right = +1, correct left = -1, no correct (error) = 0
                if df.Hit[_ - k] == 1 and df.Choice[_ - k] == 1:
                    r_plus.append(1)
                elif df.Hit[_ - k] == 1 and df.Choice[_ - k] == 0:
                    r_plus.append(-1)
                elif df.Hit[_ - k] == 0:
                    r_plus.append(0)

        return r_minus, r_plus

    design_matrix = pd.DataFrame()  # Create empty DataFrame to store previous choices

    for _ in reversed(range(1, k + 1)):
        print(f'Getting choice history of trial lag {_}')
        r_minus, r_plus = get_choice_history(df, _)
        design_matrix['Rminus' + str(_)] = r_minus
        design_matrix['Rplus' + str(_)] = r_plus

    # Reorder exog columns, so I can split later in half the params for plotting r+ or r-
    r_minus_columns = ['Rminus' + str(_) for _ in reversed(range(1, k + 1))]
    r_plus_columns = ['Rplus' + str(_) for _ in reversed(range(1, k + 1))]
    design_matrix = design_matrix[r_minus_columns + r_plus_columns]

    return design_matrix


def make_session_index_dm(df, column='Date'):
    """
    # Make a design matrix in which there are as many columns as unique dates. Then, for each column, there is a 1 if
    the trial belongs to that session and a 0 otherwise
    :param df: Input DataFrame
    :param column: Column of the DataFrame that contains the dates
    :return: Design matrix
    """
    dates = df[column].unique()
    design_matrix = np.zeros((len(df), len(dates)), dtype=int)
    for i, date in enumerate(dates):
        design_matrix[df[column] == date, i] = 1
    design_matrix = pd.DataFrame(design_matrix)
    return design_matrix


def make_ild_dm(df):
    """
    Make a design matrix with the net ILDs. There is a column for each absolute, unique ILD value (except 0). It
    transforms the nominal ILD into ILD net magnitude (2, 4, 8 , 70 dB) that take values +1, 0 or -1. In each trial,
    only one of these regressors is non-zero.
    When separating the stimuli S_k =  nominal_ILD + residuals and give a separate beta for the nominal and for each
    residual frame, you are somehow assuming that the impact of the nominal ILD grows linearly with ILD. But this is
    probably not the case. Particularly if spanning a range from ILD 0 to 70 dB. One simple way to not assume anything
    about how the impact of the stimuli grows with ILD is to define separate regressors for each absolute value of the
    ILD, that is 2, 4, 8 and 70. Each of this ILDs will define a regressor e.g. ILD_8 =  +1 (if ILD was +8 dB), -1 (if
    ILD was -8 db) and 0 (if ILD was other than +- 8dB). This way, you should be able to include ALL stimuli in the
    analysis (maximum evidence too).
    :param df:
    :return: Design matrix
    """
    ilds = df.ILD.astype('int')
    net_ilds = np.sort(df.ILD.abs().unique().astype('int'))[1:]
    design_matrix = np.zeros((len(df), len(net_ilds)), dtype=int)
    # columns = [str(_) for _ in net_ilds]
    design_matrix = pd.DataFrame(design_matrix, columns=net_ilds)
    for i, ild in enumerate(ilds):
        if ild != 0:
            design_matrix.loc[i, abs(ild)] = np.sign(ild)
    return design_matrix


def get_shuffles_GLM(endog, exog, iterations):
    """
    Permutation test. Shuffling the endog or exog indexes is the same, so it doesn't matter. Shuffling along the columns
    of stim_strength is wrong because it breaks the temporal structure of the data. Shuffling the frames within trial
    could be an interesting test, as it preserves the overall weight of the stimulus for each trial but breaks the frame
    structure
    :param endog:
    :param exog:
    :param iterations:
    :return: shuffles (list)
    """

    shuffles = []

    for _ in range(iterations):
        # print(f'Iteration {_}/{iterations}')
        endog_shuffled = endog.sample(frac=1).reset_index(drop=True)
        # exog_shuffled = exog.sample(frac=1).reset_index(drop=True)
        model_shuffled = sm.GLM(endog_shuffled, exog,  # Shuffled choices
                                family=sm.families.Binomial(), missing='drop')  # GLM with Binomial family and Logit link
        # model_shuffled = sm.GLM(endog, exog_shuffled,  # Shuffled stim_strength
        #                         family=sm.families.Binomial())  # GLM with Binomial family and Logit link
        results_shuffled = model_shuffled.fit()
        params_shuffled = results_shuffled.params
        shuffles.append(params_shuffled)

    return shuffles