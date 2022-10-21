import os
import pandas as pd
from glue_sessions.glue_sessions import update_glued_sessions


def glue_intersessions(protocol='stage_training_v3', experiment='2AFC_3', to_csv=False):
    """
    Concatenate all intersession .csv files from each animal into a single .csv file
    :param to_csv: True for saving the output DataFrame, default is False (do not save)
    :return: DataFrame with all the intersession concatenated
    """

    # Update glued sessions first
    update_glued_sessions(protocol=protocol, experiment=experiment)

    path = '/home/alexis/PycharmProjects/intersession/' + experiment + '/'
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



