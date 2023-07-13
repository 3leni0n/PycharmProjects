########################################################################################################################

# NOTE: The next 2 functions could be merged in a single one that could take as an argument if adding to drug data to the
# glued sessions or the intersessions (most code is copied/pasted)

def add_drug_data_to_intersession(path_drug='/home/alexis/Descargas/Mouse injections MK801.csv',
                                  to_csv=False):
    """
    Takes all intersession' .csv files from an experiment (batch) and a drug' .csv paths as inputs and adds the drug data
    to a new 'Drug' column to the intersession .csv to the corresponding dates (otherwise it adds nans) and saves the
    updated intersession .csv file
    :param path_drug: path of the drug data .csv. First column is 'Date' in format 'YYYY-MM-DD', subsequent columns are
    mouse number and the injection they received ('saline', 'drug' or 'rest')
    :return: intersession DataFrame with a 'Drug' column added to the end
    """

    folder_intersessions = '/home/alexis/PycharmProjects/intersession/2AFC_2/'  # Where the data for all animals is
    intersessions = os.listdir(folder_intersessions)  # List experiments
    intersessions.sort()  # Sort them by name

    for j in intersessions:

        path_intersession = folder_intersessions + j
        print(path_intersession)
        df_intersession = pd.read_csv(path_intersession)
        df_drug = pd.read_csv(path_drug)
        animal = df_intersession.Subject.unique()[0].astype('str')
        drug = []

        for i in range(len(df_intersession.Dates)):
            if len(df_drug.Date.str.contains(df_intersession.Dates[i]).unique()) == 2:
                drug.append(df_drug[df_drug.Date == df_intersession.Dates[i]][animal].values[0])
            else:
                drug.append(np.nan)

        drug = pd.Series(drug)
        df_intersession['Drug'] = drug

        if to_csv:
            df_intersession.to_csv('/home/alexis/PycharmProjects/intersession/2AFC_2/' + animal + '_intersession.csv',
                                   index=False)

    return df_intersession


def add_drug_data_to_glued_sessions(path_drug='/home/alexis/Descargas/Mouse injections MK801.csv',
                                    to_csv=False):
    """
    Takes all intersession' .csv files from an experiment (batch) and a drug' .csv paths as inputs and adds the drug data
    to a new 'Drug' column to the intersession .csv to the corresponding dates (otherwise it adds nans) and saves the
    updated intersession .csv file
    :param path_drug: path of the drug data .csv. First column is 'Date' in format 'YYYY-MM-DD', subsequent columns are
    mouse number and the injection they received ('saline', 'drug' or 'rest')
    :return: intersession DataFrame with a 'Drug' column added to the end
    """

    folder_glued_sessions = '/home/alexis/PycharmProjects/glue_sessions/2AFC_2/'  # Where the data for all animals is
    glued_sessions = os.listdir(folder_glued_sessions)  # List experiments
    glued_sessions.sort()  # Sort them by name

    for j in glued_sessions:

        path_glued_sessions = folder_glued_sessions + j
        print(path_glued_sessions)
        df_sessions = pd.read_csv(path_glued_sessions)
        df_drug = pd.read_csv(path_drug)
        animal = df_sessions.Subject.unique()[0].astype('str')
        drug = []

        for i in range(len(df_sessions.Date)):
            if len(df_drug.Date.str.contains(df_sessions.Date[i]).unique()) == 2:
                drug.append(df_drug[df_drug.Date == df_sessions.Date[i]][animal].values[0])
            else:
                drug.append(np.nan)

        drug = pd.Series(drug)
        df_sessions['Drug'] = drug

        if to_csv:
            df_sessions.to_csv('/home/alexis/PycharmProjects/glue_sessions/2AFC_2/' + animal + '.csv', index=False)

    return df_sessions


def add_drug_data(data_type='intersession', to_csv=False):
    """
    Takes all session/intersession .csv files from an experiment (batch) and a drug' .csv paths as inputs and adds
    the drug data to a new 'Drug' column to the session/intersession .csv to the corresponding dates (otherwise it adds
    nans) and saves the updated intersession .csv file
    mouse number and the injection they received ('saline', 'drug' or 'rest')
    :data_type: 'glue_sessions' or 'intersession'
    :to_csv: if True, saves the updated intersession .csv file
    :return: Last iteration DataFrame with a 'Drug' column added to the end
    """

    path_drug = Path.home() / 'PycharmProjects' / 'drugs' / 'Mouse injections MK801.csv'  # Where the drug data is
    # folder = '/home/alexis/PycharmProjects/' + data_type + '/2AFC_2/'  # Where the data for all animals is
    folder = Path.home() / 'PycharmProjects' / data_type / '2AFC_2/'  # Where the data for all animals is
    data_list = os.listdir(folder)  # List experiments
    data_list.sort()  # Sort them by name
    data_list = [i for i in data_list if '_corrupted_sessions' not in i]  # Remove '_corrupted_sessions'.csv files
    df_drug = pd.read_csv(path_drug)

    for j in data_list:

        # path = folder + j
        path = folder / j
        print(path)
        df = pd.read_csv(path)
        animal = df.Subject.unique()[0].astype('str')
        drug = []

        if data_type == 'glue_sessions':  # Trial data
            dates = df.Date
            file_extension = '.csv'
        elif data_type == 'intersession':  # Intersession data
            dates = df.Dates
            file_extension = '_intersession.csv'

        for i in range(len(df)):
            if len(df_drug.Date.str.contains(dates[i]).unique()) == 2:
                drug.append(df_drug[df_drug.Date == dates[i]][animal].values[0])
            else:
                drug.append(np.nan)

        drug = pd.Series(drug)
        df['Drug'] = drug

        if to_csv:
            # df.to_csv('/home/alexis/PycharmProjects/' + data_type + '/2AFC_2/' + animal + '.csv', index=False)
            df.to_csv(folder / (animal + file_extension), index=False)

    return df

# To debug:
# intersession_within_animal('/home/alexis/PycharmProjects/glue_sessions/2AFC_2/325.csv', to_csv=True, send_slack=False)
# do_intersessions(protocol='stage_training_v2', experiment='2AFC_2', to_csv=True, send_slack=False)
df = add_drug_data(data_type='glue_sessions', to_csv=False)