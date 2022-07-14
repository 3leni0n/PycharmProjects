import os
import pandas as pd


def glue_intersessions(to_csv=False):
    """
    Concatenate all intersession .csv files from each animal into a single .csv file
    :param to_csv: True for saving the output DataFrame, default is False (do not save)
    :return: DataFrame with all the intersession concatenated
    """

    path = '/home/alexis/PycharmProjects/intersession/'
    intersessions = os.listdir(path)  # Get list of
    intersessions.sort()
    intersessions = [x for x in intersessions if x.endswith('.csv')]  # Get rid of non csv files

    df = pd.DataFrame()

    for i in range(len(intersessions)):
        df_intersession = pd.read_csv(path + intersessions[i])
        df = pd.concat([df, df_intersession])

    if to_csv:
        df.to_csv(path + 'all_intersessions' + '.csv', index=False)  # index=False to avoid the 'Unmmaed: 0' column

    return df



