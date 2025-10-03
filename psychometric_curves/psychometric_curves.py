import warnings
warnings.filterwarnings('ignore')
import time
from pathlib import Path
import os
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from my_fun.my_fun import get_experiment, get_animal, compute_psych_curve, timer, filter_drug_sessions
from parse.parse_v2 import parse_v2

# Aesthetic parameters
# sns.set_theme()
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')


# Good animals for psychometric curves
# experiment = '2AFC_X'
# if experiment == '2AFC_2':
#     animals = ['325', '327', '329', '330', '332', '333', '335', '337']
# elif experiment == '2AFC_3':
#     animals = ['419', '420', '422', '616', '619', '623']


def plot_pc(experiment='2AFC_6', animal=None, kind='prob_right', drug=None, save=False, format='png', transparent=False):
    """Plot psychometric curve
    :param experiment: str, name of the experiment
    :param animal: str, animal name
    :param kind: str, 'prob_right' or 'prob_rep'
    :param save: bool, whether to save the figure
    :param format: str, file format to save the figure
    :param transparent: bool, whether to save the figure with a transparent background
    :return: psych_curve object with the fitted parameters and data
    """

    # Use recursion to handle multiple animals
    if isinstance(animal, list):
        psych_curves = []
        for a in animal:
            psych_curves.append(plot_pc(experiment=experiment, animal=a, kind=kind,
                                   drug=drug, save=save, format=format, transparent=transparent))
        return psych_curves

    # Get the path to the data
    experiment, folder_in = get_experiment(experiment)
    animal = get_animal(experiment=experiment, path_session='glue_sessions', animal=animal)
    folder_in = Path(folder_in / animal).with_suffix('.csv')

    # Load behavioral data
    df = pd.read_csv(folder_in)

    # Load intersession data
    path_intersession = Path.home() / 'PycharmProjects' / 'intersession' / experiment / (str(int(animal)) + '_intersession.csv')
    # str(int(animal)) to remove the 0 padding in ID
    df_intersession = pd.read_csv(path_intersession)

    # Filter trials
    df = df[df.P > 0]  # Only those sessions with ilds
    # Only sessions with accuracy > X threshold?
    # try:
    #     df = df[df.Drug.isnull()]  # Remove drug experimental sessions
    # except AttributeError: # As 24.05.2023 only batch 2 has drug data. Need to reparse batch 3 to add Drug column
    #     pass

    if drug is None:
        df = df[~df.Drug.isin([0, 1])]
    elif drug in [0, 1]:
        df = filter_drug_sessions(df)
        df = df[df.Drug == drug]

    # Compute psychometric curve(s)
    n_points = 100
    # evidences = np.sort(df.evidence.unique())  # Pilot batch
    ilds = np.sort(df.ILD.unique())

    # Plot psychometric curves
    plt.figure(constrained_layout=True)

    fontsize = 'medium'

    if kind == 'prob_right':

        # Compute left-right psychometric curve
        # psych_curve = compute_psych_curve(df.Evidence, df.Choice)  # Pilot batch
        psych_curve = compute_psych_curve(df.ILD, df.Choice, n_points)  # No need to filter out the misses

        # Move extreme datapoints closer to the center to zoom in
        psych_curve.xdata[0] = -20
        psych_curve.xdata[-1] = 20

        # Plot params
        color = 'tab:orange'
        xlabel = 'Stimulus ILD (dB)'
        ylabel = 'Prob. choose right'

        # Annotation params
        lower_lapse = "LR_R="
        upper_lapse = "LR_L="
        xy = (psych_curve.xdata[0], 1)
        xytext = (psych_curve.xdata[0], 1)
        va = 'top'
        ha = 'left'
        loc = 'lower right'

        filename = f'{animal}_PC_prob_right.{format}'

    elif kind == 'prob_rep':

        # Compute rep-alt psychometric curve
        # psych_curve_rep = compute_psych_curve(df.EviRep, df.RepChoice)  # Pilot batch
        psych_curve = compute_psych_curve(df.ILDRep, df.RepChoice, n_points)

        # Move extreme datapoints closer to the center to zoom in
        psych_curve.xdata[0] = -20
        psych_curve.xdata[-1] = 20

        # Plot params
        color = 'tab:brown'
        xlabel = 'Repeating stimulus ILD (dB)'
        ylabel = 'Prob. choose repeat'

        # Annotate params
        lower_lapse = "LR_Rep="
        upper_lapse = "LR_Alt="
        xy = (psych_curve.xdata[-1], 0)
        xytext = (psych_curve.xdata[-1], 0)
        va = 'bottom'
        ha = 'right'
        loc = 'upper left'

        filename = f'{animal}_PC_prob_rep.{format}'

    # Plot psychometric curve and errorbars
    x = np.linspace(np.min(ilds), np.max(ilds), n_points)
    plt.plot(x, psych_curve.fit, color=color, mfc=color, label='')

    plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color=color,
                 fmt='o', mfc=color)

    sensitivity, bias, lr_lower, lr_upper = psych_curve.params
    plt.annotate('S=' + str(round(sensitivity, 2)) + '\n' +  # Sensitivity
                 'B=' + str(round(bias, 2)) + '\n' +  # Bias
                 lower_lapse + str(round(lr_lower, 2)) + '\n' +  # Upper lapse rate
                 upper_lapse + str(round(lr_upper, 2)),  # Lower lapse rate
                 xy=xy, xytext=xytext, color=color,
                 va=va, ha=ha, fontsize=fontsize)

    plt.title(f'Mouse {df.Setup.unique()[0]}, {len(df)} trials')
    plt.axhline(0.5, color='tab:gray', ls='--')
    plt.axvline(0, color='tab:gray', ls='--')

    # # Get fits for bias = 0 and lapses = 0
    # # fit = b + (1 - b - p) / (1 + np.exp(-k * (np.linspace(np.min(x), np.max(x), n_points) - x0)))  # PC function
    # fit_bias0 = lr_lower + (1 - lr_lower - lr_upper) / (1 + np.exp(- sensitivity * (np.linspace(np.min(ilds), np.max(ilds), n_points) - 0)))
    # # plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), fit_bias0, color='tab:olive', mfc='tab:olive', ls=':', label='fit|B=0')
    # pc0_bias0 = lr_lower + (1 - lr_lower - lr_upper) / 2  # Value of the PC for x = 0 when bias = 0
    # fit_lapses0 = 0 + (1 - 0 - 0) / (1 + np.exp(- sensitivity * (np.linspace(np.min(ilds), np.max(ilds), n_points) - bias)))
    # # plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), fit_lapses0, color='tab:cyan', mfc='tab:cyan', ls=':', label='fit|LR=0')
    # # plt.axhline(pc0_bias0, color='tab:blue', ls=':', label='y(x=0)|B=0')
    # pc0_lapses0 = 1 / (1 + np.exp(sensitivity * bias))  # Value of the PC for x = 0 when lapses = 0
    # # plt.axhline(pc0_lapses0, color='tab:orange', ls=':', label='y(x=0)|LR=0')

    plt.xlim([psych_curve.xdata[0] - 1, psych_curve.xdata[-1] + 1])  # To chop the extreme values
    ilds[0] = psych_curve.xdata[0]
    ilds[-1] = psych_curve.xdata[-1]
    plt.xticks(ilds)
    plt.gca().set_xticklabels(['-70', '-8', '', '', '0', '', '', '8', '70'])
    # plt.ylim([-0.025, 1.025])
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.despine()

    if save:
        folder_out = Path.home() / 'OneDrive' / 'Imágenes' / 'Figures' / 'psych curves'
        if not folder_out.exists():
            folder_out.mkdir(parents=True, exist_ok=True)
        os.chdir(folder_out)
        plt.savefig(Path(folder_out / filename), format=format, transparent=transparent)
        plt.close()

    # return psych_curve, pc0_bias0, pc0_lapses0
    # return psych_curve, fit_bias0, fit_lapses0
    return psych_curve


def plot_mean_pc(experiment='2AFC_6', animals=['014', '016', '017', '020', '021', '022', '023', '024', '025'], save=True,
                 kind='prob_right', drug=np.nan, format='png', transparent=False):
    """Plot psychometric curves across animals
    :param experiment: str, name of the experiment
    :param animals: list, list of animal names
    :param kind: str, 'prob_right' or 'prob_rep'
    :param save: bool, whether to save the figure
    :param format: str, file format to save the figure
    :param transparent: bool, whether to save the figure with a transparent background
    :return: params: array of psychometric curve parameters for each animal
    """

    # Get the path to the data
    experiment, folder_in = get_experiment(experiment)

    fit = []
    fit_error = []
    params = []
    xdata = []
    ydata = []
    trials = 0
    animals_removed = 0
    n_points = 100

    plt.figure(constrained_layout=True)
    fontsize = 'medium'

    for i in range(len(animals)):

        df = pd.read_csv(Path(folder_in / animals[i]).with_suffix('.csv'))
        df = df[df.P > 0]  # Only those sessions with ilds

        if drug is None:
            df = df[~df.Drug.isin([0, 1])]
        elif drug in [0, 1]:
            df = filter_drug_sessions(df)
            df = df[df.Drug == drug]

        ilds = np.sort(df.ILD.unique())
        if len(ilds) != 9:
            print(f'Animal {animals[i]} has {len(ilds)} ILDs, not 9. Need more trials. Skipping...')
            animals_removed += 1
            continue

        if kind == 'prob_right':
            psych_curve = compute_psych_curve(df.ILD, df.Choice, n_points)
        elif kind == 'prob_rep':
            psych_curve = compute_psych_curve(df.ILDRep, df.RepChoice, n_points)

        plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve.fit, color='tab:grey', marker=None,
                 mfc='none', mec='none', mew=0, ms=0, alpha=0.25)
        # mec='none, mew=0 and ms=0 to not plot markers in individual animals
        # plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color='tab:grey', fmt='o',
        #              mfc='none', alpha=0.25)  # Don't plot the errorbars in individual animals

        fit.append(psych_curve.fit)
        fit_error.append(psych_curve.fit_error)
        params.append(psych_curve.params)
        xdata.append(psych_curve.xdata)
        ydata.append(psych_curve.ydata)
        trials += len(df)

    xdata = np.array(xdata).flatten()
    ydata = np.array(ydata).flatten()
    psych_curve = compute_psych_curve(xdata, ydata, n_points)

    # Move extreme datapoints closer to the center to zoom in
    psych_curve.xdata[0] = -20
    psych_curve.xdata[-1] = 20

    if kind == 'prob_right':

        # Plot params
        color = 'tab:orange'
        label = 'Prob. right'
        xlabel = 'Stimulus ILD (dB)'
        ylabel = 'Prob. choose right'

        # Annotation params
        lower_lapse = "LR_R="
        upper_lapse = "LR_L="
        xy = (psych_curve.xdata[0], 1)
        xytext = (psych_curve.xdata[0], 1)
        va = 'top'
        ha = 'left'

        filename = f'{experiment}_PC_prob_right.{format}'

    elif kind == 'prob_rep':

        # Plot params
        color = 'tab:brown'
        label = 'Prob. repeat'
        xlabel = 'Repeating stimulus ILD (dB)'
        ylabel = 'Prob. choose repeat'

        # Annotate params
        lower_lapse = "LR_Rep="
        upper_lapse = "LR_Alt="
        xy = (psych_curve.xdata[-1], 0)
        xytext = (psych_curve.xdata[-1], 0)
        va = 'bottom'
        ha = 'right'

        filename = f'{experiment}_PC_prob_rep.{format}'

    plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve.fit, color=color, mfc=color,
             label=label)

    plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color=color, fmt='o',
                 mfc=color)

    plt.axhline(0.5, color='tab:gray', ls='--')
    plt.axvline(0., color='tab:gray', ls='--')
    plt.title(f'N={len(animals) - animals_removed}, {trials} trials')

    plt.xlim([psych_curve.xdata[0] - 1, psych_curve.xdata[-1] + 1])  # To chop the extreme values
    ilds[0] = psych_curve.xdata[0]
    ilds[-1] = psych_curve.xdata[-1]
    plt.xticks(ilds)
    plt.gca().set_xticklabels(['-70', '-8', '', '', '0', '', '', '8', '70'])
    # plt.ylim([-0.025, 1.025])
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.despine()

    sensitivity, bias, lr_lower, lr_upper = psych_curve.params
    plt.annotate('S=' + str(round(sensitivity, 2)) + '\n' +  # Sensitivity
                 'B=' + str(round(bias, 2)) + '\n' +  # Bias
                 lower_lapse + str(round(lr_lower, 2)) + '\n' +  # Upper lapse rate
                 upper_lapse + str(round(lr_upper, 2)),  # Lower lapse rate
                 xy=xy, xytext=xytext, color=color,
                 va=va, ha=ha, fontsize=fontsize)

    if save:
        folder_out = Path.home() / 'OneDrive' / 'Imágenes' / 'Figures' / 'psych curves'
        if not folder_out.exists():
            folder_out.mkdir(parents=True, exist_ok=True)
        os.chdir(folder_out)
        plt.savefig(Path(folder_out / filename), format=format, transparent=transparent)
        plt.close()

    params = np.array(params)
    return psych_curve, params

def plot_pc_drug(experiment='2AFC_6', animal='020', kind='prob_right'):
    """
    Plot psychometric curves across animals for saline and drug conditions
    :param experiment: str, name of the experiment
    :param animals: list, list of animal names
    :param kind: str, 'prob_right' or 'prob_rep'
    :return:
    """

    if kind == 'prob_right':
        color = 'tab:orange'
        xlabel = 'Stimulus ILD (dB)'
        ylabel = 'Prob. choose right'
        loc = 'upper center'
        lower_lapse = "LR_R="
        upper_lapse = "LR_L="
        columns = ['sensitivity', 'bias', 'lr_right', 'lr_left', 'drug']
    elif kind == 'prob_rep':
        color = 'tab:brown'
        xlabel = 'Repeating stimulus ILD (dB)'
        ylabel = 'Prob. choose repeat'
        loc = 'lower center'
        columns = ['sensitivity', 'bias', 'lr_rep', 'lr_alt', 'drug']
        lower_lapse = "LR_Rep="
        upper_lapse = "LR_Alt="

    df_params = pd.DataFrame(columns=columns)
    plt.figure(constrained_layout=True)
    fontsize = 'medium'

    for drug in range(2):

        psych_curve = plot_pc(experiment=experiment, animal=animal, kind='prob_right', drug=drug, save=False, format='png',
                              transparent=False)

        params = psych_curve.params
        plt.close()

        if drug == 0:
            label = 'saline'

            # Annotation params
            xy = (psych_curve.xdata[0], 1)
            xytext = (psych_curve.xdata[0], 1)
            va = 'top'
            ha = 'left'
        elif drug == 1:
            color = 'tab:pink'
            label = 'drug'  # (MK-801)

            # Annotate params
            xy = (psych_curve.xdata[-1], 0)
            xytext = (psych_curve.xdata[-1], 0)
            va = 'bottom'
            ha = 'right'

        # Add drug column to params
        # params = np.hstack((params, np.full((params.shape[0], 1), drug, dtype=int)))
        params = params + [drug]
        # Add params to DataFrame
        df_params = pd.concat([df_params, pd.DataFrame([params], columns=columns)], ignore_index=True)

        # Plot the PCs in the same figure
        ilds = psych_curve.xdata  # ILDs
        n_points = 100  # Number of points to plot the psychometric curve

        plt.plot(np.linspace(-70, 70, n_points), psych_curve.fit, color=color, mfc=color, label=label)
        plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color=color, fmt='o', mfc=color)

        sensitivity, bias, lr_lower, lr_upper = psych_curve.params
        plt.annotate('S=' + str(round(sensitivity, 2)) + '\n' +  # Sensitivity
                     'B=' + str(round(bias, 2)) + '\n' +  # Bias
                     lower_lapse + str(round(lr_lower, 2)) + '\n' +  # Upper lapse rate
                     upper_lapse + str(round(lr_upper, 2)),  # Lower lapse rate
                     xy=xy, xytext=xytext, color=color,
                     va=va, ha=ha, fontsize=fontsize)

    plt.axhline(0.5, color='tab:gray', ls='--')
    plt.axvline(0., color='tab:gray', ls='--')
    # plt.title(f'N={len(animals) - animals_removed}, {trials} trials')
    # plt.title('N=7')
    plt.xlim([psych_curve.xdata[0] - 1, psych_curve.xdata[-1] + 1])  # To chop the extreme values
    ilds[0] = psych_curve.xdata[0]
    ilds[-1] = psych_curve.xdata[-1]
    plt.xticks(ilds)
    plt.gca().set_xticklabels(['-70', '-8', '', '', '0', '', '', '8', '70'])
    # plt.ylim([-0.025, 1.025])
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.despine()
    plt.legend(frameon=False, loc=loc)

    return df_params


def plot_mean_pc_drug(experiment='2AFC_6', animals=['014', '016', '017', '020', '021', '022', '023', '024', '025'],
                      kind='prob_right'):
    """
    Plot psychometric curves across animals for saline and drug conditions
    :param experiment: str, name of the experiment
    :param animals: list, list of animal names
    :param kind: str, 'prob_right' or 'prob_rep'
    :return:
    """

    if kind == 'prob_right':
        color = 'tab:orange'
        xlabel = 'Stimulus ILD (dB)'
        ylabel = 'Prob. choose right'
        loc = 'upper center'
        lower_lapse = "LR_R="
        upper_lapse = "LR_L="
        columns = ['sensitivity', 'bias', 'lr_right', 'lr_left', 'drug']
    elif kind == 'prob_rep':
        color = 'tab:brown'
        xlabel = 'Repeating stimulus ILD (dB)'
        ylabel = 'Prob. choose repeat'
        loc = 'lower center'
        columns = ['sensitivity', 'bias', 'lr_rep', 'lr_alt', 'drug']
        lower_lapse = "LR_Rep="
        upper_lapse = "LR_Alt="

    df_params = pd.DataFrame(columns=columns)
    plt.figure(constrained_layout=True)
    fontsize = 'medium'

    for drug in range(2):

        psych_curve, params = plot_mean_pc(experiment=experiment, animals=animals, save=False, kind=kind, drug=drug,
                                                    format='png', transparent=False)
        plt.close()

        if drug == 0:
            label = 'saline'

            # Annotation params
            xy = (psych_curve.xdata[0], 1)
            xytext = (psych_curve.xdata[0], 1)
            va = 'top'
            ha = 'left'
        elif drug == 1:
            color = 'tab:pink'
            label = 'drug'  # (MK-801)

            # Annotate params
            xy = (psych_curve.xdata[-1], 0)
            xytext = (psych_curve.xdata[-1], 0)
            va = 'bottom'
            ha = 'right'

        # Add drug column to params
        params = np.hstack((params, np.full((params.shape[0], 1), drug, dtype=int)))
        # Add params to DataFrame
        df_params = pd.concat([df_params, pd.DataFrame(params, columns=columns)], ignore_index=True)

        # Plot the PCs in the same figure
        ilds = psych_curve.xdata  # ILDs
        n_points = 100  # Number of points to plot the psychometric curve

        plt.plot(np.linspace(-70, 70, n_points), psych_curve.fit, color=color, mfc=color, label=label)
        plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color=color, fmt='o', mfc=color)

        sensitivity, bias, lr_lower, lr_upper = psych_curve.params
        plt.annotate('S=' + str(round(sensitivity, 2)) + '\n' +  # Sensitivity
                     'B=' + str(round(bias, 2)) + '\n' +  # Bias
                     lower_lapse + str(round(lr_lower, 2)) + '\n' +  # Upper lapse rate
                     upper_lapse + str(round(lr_upper, 2)),  # Lower lapse rate
                     xy=xy, xytext=xytext, color=color,
                     va=va, ha=ha, fontsize=fontsize)

    plt.axhline(0.5, color='tab:gray', ls='--')
    plt.axvline(0., color='tab:gray', ls='--')
    # plt.title(f'N={len(animals) - animals_removed}, {trials} trials')
    # plt.title('N=7')
    plt.xlim([psych_curve.xdata[0] - 1, psych_curve.xdata[-1] + 1])  # To chop the extreme values
    ilds[0] = psych_curve.xdata[0]
    ilds[-1] = psych_curve.xdata[-1]
    plt.xticks(ilds)
    plt.gca().set_xticklabels(['-70', '-8', '', '', '0', '', '', '8', '70'])
    # plt.ylim([-0.025, 1.025])
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.despine()
    plt.legend(frameon=False, loc=loc)

    return df_params

########################################################################################################################

# Across batches

def plot_pc_across_batches(experiments=['2AFC_2', '2AFC_3'], animals=None,
                           save=False, kind='prob_right', format='svg', transparent=False):

    fit = []
    fit_error = []
    params = []
    xdata = []
    ydata = []
    trials = 0

    n_points = 100

    plt.figure(constrained_layout=True)

    for k in range(len(experiments)):

        # Get the path to the data
        # experiment = get_experiment(experiment[experiments[k]])
        experiment = experiments[k]
        folder_in = Path.home() / 'PycharmProjects' / 'glue_sessions' / experiment

        if experiment == '2AFC_2':
            animals = ['325', '327', '329', '330', '332', '333', '335', '337']
            n_animals_batch2 = len(animals)
        elif experiment == '2AFC_3':
            animals = ['419', '420', '422', '616', '619', '623']
            n_animals_batch3 = len(animals)

        ####################################################################################################################

        for i in range(len(animals)):
            df = pd.read_csv(Path(folder_in / animals[i]).with_suffix('.csv'))
            # df = df[df.P > 0]  # Only those sessions with ilds
            ilds = np.sort(df.ILD.unique())

            if kind == 'prob_right':
                psych_curve = compute_psych_curve(df.ILD, df.Choice, n_points)
            elif kind == 'prob_rep':
                psych_curve = compute_psych_curve(df.ILDRep, df.RepChoice, n_points)

            plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve.fit, color='tab:grey', marker=None,
                     mfc='none', mec='none', mew=0, ms=0, alpha=0.25)
            # mec='none, mew=0 and ms=0 to not plot markers in individual animals
            # plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color='tab:grey', fmt='o',
            #              mfc='none', alpha=0.25)  # Don't plot the errorbars in individual animals

            fit.append(psych_curve.fit)
            fit_error.append(psych_curve.fit_error)
            params.append(psych_curve.params)
            xdata.append(psych_curve.xdata)
            ydata.append(psych_curve.ydata)
            trials += len(df)

    xdata = np.array(xdata).flatten()
    ydata = np.array(ydata).flatten()
    psych_curve = compute_psych_curve(xdata, ydata, n_points)

    n_animals = n_animals_batch2 + n_animals_batch3

    if kind == 'prob_right':

        # Plot params
        color = 'tab:orange'
        label = 'Prob. right'
        xlabel = 'Stimulus ILD (dB)'
        ylabel = 'Prob. choose right'

        # Annotation params
        lower_lapse = "LR_R="
        upper_lapse = "LR_L="
        # xy = (ilds[0], 1)
        # xytext = (ilds[0], 1)
        xy = (-20, 1)
        xytext = (-20, 1)
        va = 'top'
        ha = 'left'
        fontsize = 'medium'

        filename = ' PC_prob_right_all_animals' + '.' + format

    elif kind == 'prob_rep':

        # Plot params
        color = 'tab:brown'
        label = 'Prob. repeat'
        xlabel = 'Repeating stimulus ILD (dB)'
        ylabel = 'Prob. choose repeat'

        # Annotate params
        lower_lapse = "LR_Rep="
        upper_lapse = "LR_Alt="
        # xy = (ilds[-1], 0)
        # xytext = (ilds[-1], 0)
        xy = (20, 0)
        xytext = (20, 0)
        color = 'tab:brown'
        va = 'bottom'
        ha = 'right'
        fontsize = 'medium'

        filename = ' PC_prob_rep_all_animals' + '.' + format

    plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve.fit, color=color, mfc=color,
             label=label)

    # Move extreme datapoints closer to the center to zoom in
    psych_curve.xdata[0] = -20
    psych_curve.xdata[-1] = 20

    plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color=color, fmt='o',
                 mfc=color)

    # plt.xscale('symlog', linthreshx=20)  # Set symmetric logarithmic spacing to zoom in the middle
    # plt.minorticks_off()  # Remove minor ticks
    plt.axhline(0.5, color='tab:gray', ls='--')
    plt.axvline(0., color='tab:gray', ls='--')
    # plt.title(f'N={len(animals)}, {trials} trials')
    plt.title(f'N={n_animals}, {trials} trials')
    # plt.title(f'N={len(df.Setup.unique())}, {len(df)} trials')
    # plt.xlim([ilds[0] - 7, ilds[-1] + 7])
    plt.xlim([-21, 21])  # To chop the extreme values
    ilds[0] = -20
    ilds[-1] = 20
    plt.xticks(ilds)
    plt.gca().set_xticklabels(['-70', '-8', '', '', '0', '', '', '8', '70'])
    plt.ylim([-0.025, 1.025])
    plt.yticks([0, 0.5, 1])
    # plt.legend(loc='lower center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    sensitivity, bias, lr_lower, lr_upper = psych_curve.params
    plt.annotate("S=" + str(round(sensitivity, 2)) + "\n" +  # Sensitivity
                 "B=" + str(round(bias, 2)) + "\n" +  # Bias
                 lower_lapse + str(round(lr_lower, 2)) + "\n" +  # Upper lapse rate
                 upper_lapse + str(round(lr_upper, 2)),  # Lower lapse rate
                 xy=xy, xytext=xytext, color=color,
                 va=va, ha=ha, fontsize=fontsize)

    if save:
        folder_out = Path.home() / 'Documentos/psychometric curves/' / experiment
        if not folder_out.exists():
            folder_out.mkdir(parents=True, exist_ok=True)
        os.chdir(folder_out)
        plt.savefig(Path(folder_out / filename), format=format, transparent=transparent)
        plt.close()

    return np.array(params)


########################################################################################################################

def test_lapses(experiment='2AFC_2', animals=['325', '327', '329', '330', '332', '333', '335', '337'], kind='prob_rep',
                save=True, format='png', transparent=False):

    params = plot_pc_across_animals(experiment=experiment, animals=animals, kind=kind)
    lower_lapses = params[:, 2]
    upper_lapses = params[:, 3]
    plt.figure(constrained_layout=True)
    plt.scatter(upper_lapses, lower_lapses, color='k')
    x = np.linspace(0, 0.5)
    y = x
    plt.plot(x, y, 'r')
    plt.axis('square')
    plt.axhline(0.2, color='tab:gray', ls=':')
    plt.axvline(0.2, color='tab:gray', ls=':')
    # plt.title('Rep. vs Alt. lapses')
    # plt.xlim(0, 0.5)
    plt.xlim(0, np.round(np.max(lower_lapses), 1))
    plt.xticks(np.linspace(0, np.round(np.max(lower_lapses), 1), 3))
    # plt.ylim(0, 0.5)
    plt.ylim(0, np.round(np.max(lower_lapses), 1))
    plt.yticks(np.linspace(0, np.round(np.max(lower_lapses), 1), 3))

    if kind == 'prob_right':
        plt.xlabel('Left lapses')
        plt.ylabel('Right lapses')
    else:
        plt.xlabel('Alt. lapses')
        plt.ylabel('Rep. lapses')

    # Paired-samples t-test
    t_test = stats.ttest_rel(lower_lapses, upper_lapses)  # Two-sided
    # t_test = stats.ttest_rel(lower_lapses, upper_lapses, alternative='greater')  # One-sided (you have a priori hypothesis of
    # why one mean would be larger than the other one)

    plt.annotate('Rep. lapse = ' + str(np.round(np.mean(lower_lapses), 2)) + ' ± ' +
                 str(np.round(stats.sem(lower_lapses), 2)), xy=(0.2, 0.04), color='k', va='bottom', ha='center',
                 fontsize='xx-small')
    plt.annotate('Alt. lapse = ' + str(np.round(np.mean(upper_lapses), 2)) + ' ± ' +
                 str(np.round(stats.sem(upper_lapses), 2)),
                 xy=(0.2, 0.02), color='k', va='bottom', ha='center', fontsize='xx-small')
    plt.annotate('p = ' + str(np.round(t_test.pvalue, 4)), xy=(0.2, 0), color='k', va='bottom', ha='center',
                 fontsize='xx-small')

    if save:
        filename = kind + '_lapses' + '.' + format
        folder_out = Path.home() / 'Documentos' / 'psychometric curves' / experiment
        plt.savefig(Path(folder_out / filename), format=format, transparent=transparent)


def plot_pc_session(path, kind='prob_right', annotation_loc='upper left', color='tab:orange', label=''):

    # path = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220520-111627/332_stage_training_v2_20220520-111627.csv'
    df = parse_v2(path)

    n_points = 100
    ilds = np.sort(df.ILD.unique())

    # plt.figure(constrained_layout=True)
    fontsize = 'medium'

    if kind == 'prob_right':

        # Compute left-right psychometric curve
        psych_curve = compute_psych_curve(df.ILD, df.Choice, n_points)  # No need to filter out the misses

        # Plot params
        # color = 'tab:orange'
        # label = 'Prob. right'
        plt.xlabel('Stimulus ILD (dB)')
        plt.ylabel('Prob. choose right')

        # Annotation params
        lower_lapse = "LR_R="
        upper_lapse = "LR_L="
        filename = '_PC_prob_right'

    elif kind == 'prob_rep':

        # Compute rep-alt psychometric curve
        # psych_curve_rep = compute_psych_curve(df.EviRep, df.RepChoice)  # Pilot batch
        psych_curve = compute_psych_curve(df.ILDRep, df.RepChoice, n_points)

        # Plot params
        # color = 'tab:brown'
        # label = 'Prob. repeat'
        plt.xlabel('Repeating stimulus ILD (dB)')
        plt.ylabel('Prob. choose repeat')

        # Annotate params
        lower_lapse = "LR_Rep="
        upper_lapse = "LR_Alt="
        filename = '_PC_prob_rep'

    if annotation_loc == 'upper left':
        # xy = (ilds[0], 1)
        # xytext = (ilds[0], 1)
        xy = (-20, 1)
        xytext = (-20, 1)
        va = 'top'
        ha = 'left'

    elif annotation_loc == 'lower right':
        # xy = (ilds[-1], 0)
        # xytext = (ilds[-1], 0)
        xy = (20, 0)
        xytext = (20, 0)
        va = 'bottom'
        ha = 'right'

    # Plot psychometric curve and errorbars
    plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve.fit, color=color, mfc=color, label=label)

    # Move extreme datapoints closer to the center to zoom in
    psych_curve.xdata[0] = -20
    psych_curve.xdata[-1] = 20

    plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color=color,
                 fmt='o', mfc=color)

    sensitivity, bias, lr_lower, lr_upper = psych_curve.params
    # plt.annotate("S=" + str(round(sensitivity, 2)) + "\n" +  # Sensitivity
    #              "B=" + str(round(bias, 2)) + "\n" +  # Bias
    #              lower_lapse + str(round(lr_lower, 2)) + "\n" +  # Upper lapse rate
    #              upper_lapse + str(round(lr_upper, 2)),  # Lower lapse rate
    #              xy=xy, xytext=xytext, color=color,
    #              va=va, ha=ha, fontsize=fontsize)

    # plt.xscale('symlog', linthreshx=20)  # Set symmetric logarithmic spacing to zoom in the middle
    # plt.minorticks_off()  # Remove minor ticks
    # plt.title(f'Mouse {df.Setup.unique()[0]}, {len(df)} trials')
    # plt.title(f'Mouse {len(df.Setup.unique())}, {len(df)} trials')
    # plt.title(f'Mouse {df.Setup.unique()[0]} ({df.Date.unique()[0]})')
    plt.axhline(0.5, color='tab:gray', ls='--')
    plt.axvline(0., color='tab:gray', ls='--')
    # plt.xlim([ilds[0]-7, ilds[-1]+7]
    plt.xlim([-21, 21])  # To chop the extreme values
    ilds[0] = -20
    ilds[-1] = 20
    plt.xticks(ilds)
    plt.gca().set_xticklabels(['-70', '-8', '', '', '0', '', '', '8', '70'])
    plt.ylim([-0.025, 1.025])
    plt.yticks([0, 0.5, 1])
    # plt.legend(loc='lower center')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


def pc_session_comparison(path1, path2, kind='prob_rep', color='tab:brown', save=True, format='png', transparent=False):

    plt.figure(constrained_layout=True)
    # path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220520-111627/332_stage_training_v2_20220520-111627.csv'
    # path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220521-123227/332_stage_training_v2_20220521-123227.csv'

    df1 = parse_v2(path1)
    df2 = parse_v2(path2)

    animal1 = df1.Setup.unique()[0]
    animal2 = df2.Setup.unique()[0]
    animal = str(np.unique(animal1, animal2)[0])[2:-2]

    assert animal1 == animal2

    date1 = df1.Date.unique()[0]
    date2 = df2.Date.unique()[0]

    plot_pc_session(path1, kind=kind, annotation_loc='upper left', color='k', label='saline')
    plot_pc_session(path2, kind=kind, annotation_loc='lower right', color=color, label='drug')
    plt.title(f'Mouse {animal} ({date1} / {date2[-2:]})')
    plt.legend(loc='upper left', frameon=False, fontsize='xx-small')

    if kind == 'prob_right':
        filename = '_PC_prob_right_MK-801_'

    elif kind == 'prob_rep':
        filename = '_PC_prob_rep_MK-801_'

    if save:
        # folder_out = '/home/alexis/Documentos/psychometric curves/MK-801/' + animal + '/'
        folder_out = Path.home() / 'Documentos/psychometric curves/' + animal + '/'
        if not os.path.exists(folder_out):
            os.mkdir(folder_out)
        os.chdir(folder_out)
        plt.savefig(folder_out + animal + filename + date1 + '_' + date2, format=format, transparent=transparent)
        plt.close()


def plot_bias_vs_lapses(kind='prob_rep', save=False, format='svg', transparent=False):

    # psych_curves, pc0_bias0s, pc0_lapses0s = plot_pcs(experiment='2AFC_2',
    #                                                 animals=['325', '327', '329', '330', '332', '333', '335', '337'],
    #                                                 kind=kind,
    #                                                 save=False, format='svg', transparent=False)

    psych_curves, fits_bias0, fits_lapses0 = plot_pcs(experiment='2AFC_2',
                                                      animals=['325', '327', '329', '330', '332', '333', '335', '337'],
                                                      kind=kind, save=False, format='svg', transparent=False)

    fits_bias0 = np.array(fits_bias0)
    mean_fits_bias0 = np.mean(fits_bias0, axis=1)
    fits_lapses0 = np.array(fits_lapses0)
    mean_fits_lapses0 = np.mean(fits_lapses0, axis=1)

    plt.figure(constrained_layout=True)
    # plt.scatter(pc0_bias0s, pc0_lapses0s, color='k')
    plt.scatter(mean_fits_lapses0, mean_fits_bias0, color='k')
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    ax_min = np.min([xlim, ylim])
    ax_max = np.max([xlim, ylim])
    x = np.linspace(0, 1)
    y = x
    plt.plot(x, y, 'r')
    plt.axis('square')
    # plt.xlim(ax_min, ax_max)
    plt.xlim(0, 1)
    # plt.ylim(ax_min, ax_max)
    plt.ylim(0, 1)
    plt.gca().set_xticks([0, 0.5, 1])
    plt.gca().set_xticklabels(['0.0', '0.5', '1'])
    plt.gca().set_yticks([0, 0.5, 1])
    plt.gca().set_yticklabels(['0.0', '0.5', '1'])
    plt.axhline(0.5, color='tab:gray', ls=':')
    plt.axvline(0.5, color='tab:gray', ls=':')
    # plt.xlabel('bias = 0')
    # plt.ylabel('lapses = 0')

    if kind == 'prob_right':
        plt.xlabel('P. right due to bias')
        plt.ylabel('P. right due to lapses')
    else:
        plt.xlabel('P. rep due to bias')
        plt.ylabel('P. rep due to lapses')

    # plt.title(kind + '(x=0)')
    plt.title('Lapses vs bias impact on choose rep.')
    plt.title('')

    # Paired-samples t-test
    t_test = stats.ttest_rel(mean_fits_bias0, mean_fits_lapses0)  # Two-sided
    # t_test = stats.ttest_rel(lower_lapses, upper_lapses, alternative='greater')  # One-sided (you have a priori hypothesis of
    # why one mean would be larger than the other one)

    plt.annotate('Prob. rep lapses = ' + str(np.round(np.mean(mean_fits_bias0), 2)) + ' ± ' +
                 str(np.round(stats.sem(mean_fits_bias0), 2)), xy=(0.5, 0.10), color='k', va='bottom', ha='center',
                 fontsize='xx-small')
    plt.annotate('Prob. rep bias = ' + str(np.round(np.mean(mean_fits_lapses0), 2)) + ' ± ' +
                 str(np.round(stats.sem(mean_fits_lapses0), 2)),
                 xy=(0.5, 0.05), color='k', va='bottom', ha='center', fontsize='xx-small')
    plt.annotate('p = ' + str(np.round(t_test.pvalue, 4)), xy=(0.5, 0), color='k', va='bottom', ha='center',
                 fontsize='xx-small')

    plt.savefig('/home/alexis/Escritorio/' + 'Lapses vs bias impact on choose rep.', format='svg', transparent=True)

    if save:
        filename = kind + '_lapses_vs_bias' + '.' + format
        folder_out = Path.home() / 'Documentos' / 'psychometric curves' / experiment
        plt.savefig(Path(folder_out / filename), format=format, transparent=transparent)

    # # Annotate each dot
    # for i, txt in enumerate(animals):
    #     plt.gca().annotate(txt, (pc0_bias0s[i], pc0_lapses0s[i]))


# Debugging
# experiment = '2AFC_2'
# experiment = '2AFC_3'
# # animal = '333'
# animals = ['325', '327', '329', '330', '332', '333', '335', '337']  # Bach 2 (with ILDs) -326, -334
# animals = ['419', '420', '422', '616', '619', '623']  # Batch 3 (with ILDs)  -617, -620
# # animals = ['332', '333', '337']  # Drug experiments
# kind = 'prob_rep'
# save = False
# format = 'svg'
# transparent = True


# Psych curves for drug data
def plot_pc_across_animals_drug(experiment='2AFC_2', animals=['332', '333', '337'],
                           save=False, kind='prob_rep', format='svg', transparent=False):

    time_start = time.time()

    if experiment is None:

        # folder_in = '/home/alexis/PycharmProjects/glue_sessions/'  # Where the data for all animals is
        folder_in = Path.home() / 'PycharmProjects/glue_sessions'  # Where the data for all animals is
        experiments = os.listdir(folder_in)  # List experiments
        experiments.sort()  # Sort them by name
        # experiments = [x for x in experiments if os.path.isdir(folder_in + x)]  # Get rid of non folders
        experiments = [x for x in experiments if os.path.isdir(folder_in / x)]  # Get rid of non folders

        try:
            experiments.remove('__pycache__')  # Pycharm's archive
        except ValueError:
            pass

        print('Experiments: ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')

    # folder_in = '/home/alexis/PycharmProjects/glue_sessions/' + experiment + '/'  # Where the data for all animals is
    folder_in = Path.home() / 'PycharmProjects/glue_sessions' / experiment  # Where the data for all animals is

    fit = []
    fit_error = []
    params = []
    xdata = []
    ydata = []
    trials = 0

    n_points = 100
    # plt.figure(constrained_layout=True)

    for i in range(len(animals)):
        print(folder_in + animals[i] + '.csv')
        df = pd.read_csv(folder_in + animals[i] + '.csv')
        df = df[df.P > 0]
        ilds = np.sort(df.ILD.unique())

        drug = 'MK801'
        df = df[df.Drug == drug]
        # df = df[df.Drug == drug]
        # df = df[df.Drug == drug]
        if drug == 'MK801':
            color = 'tab:pink'
            alpha = 1
            ls = '-'
        else:
            color = 'tab:gray'
            alpha = 0.5
        label=drug

        df.drop(index=df[(df.Date == '2022-05-25') & (df.Setup==337)].index, inplace=True)
        df.drop(index=df[(df.Date == '2022-05-24') & (df.Setup==337)].index, inplace=True)
        df.drop(index=df[(df.Date == '2022-05-26') & (df.Setup==332)].index, inplace=True)
        df.drop(index=df[(df.Date == '2022-05-27') & (df.Setup==333)].index, inplace=True)
        df.drop(index=df[(df.Date == '2022-05-31') & (df.Setup==333)].index, inplace=True)

        if kind == 'prob_right':
            psych_curve = compute_psych_curve(df.ILD, df.Choice, n_points)
        elif kind == 'prob_rep':
            psych_curve = compute_psych_curve(df.ILDRep, df.RepChoice, n_points)

        # plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve.fit, color='tab:grey', marker=None,
                 # mfc='none', mec='none', mew=0, ms=0, alpha=0.25)
        # mec='none, mew=0 and ms=0 to not plot markers in individual animals
        # plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color='tab:grey', fmt='o',
        #              mfc='none', alpha=0.25)  # Don't plot the errorbars in individual animals

        fit.append(psych_curve.fit)
        fit_error.append(psych_curve.fit_error)
        params.append(psych_curve.params)
        xdata.append(psych_curve.xdata)
        ydata.append(psych_curve.ydata)
        trials += len(df)

    xdata = np.array(xdata).flatten()
    ydata = np.array(ydata).flatten()
    psych_curve = compute_psych_curve(xdata, ydata, n_points)

    if kind == 'prob_right':

        # Plot params
        # color = 'tab:orange'
        # label = 'Prob. right'
        xlabel = 'Stimulus ILD (dB)'
        ylabel = 'Prob. choose right'

        # Annotation params
        lower_lapse = "LR_R="
        upper_lapse = "LR_L="
        # xy = (ilds[0], 1)
        # xytext = (ilds[0], 1)
        xy = (-20, 1)
        xytext = (-20, 1)
        va = 'top'
        ha = 'left'
        fontsize = 'medium'

        filename = ' PC_prob_right_all_animals'

    elif kind == 'prob_rep':

        # Plot params
        # color = 'tab:brown'
        # label = 'Prob. repeat'
        xlabel = 'Repeating stimulus ILD (dB)'
        ylabel = 'Prob. choose repeat'

        # Annotate params
        lower_lapse = "LR_Rep="
        upper_lapse = "LR_Alt="
        # xy = (ilds[-1], 0)
        # xytext = (ilds[-1], 0)
        xy = (20, 0)
        xytext = (20, 0)
        va = 'bottom'
        ha = 'right'
        fontsize = 'medium'

        filename = ' PC_prob_rep_all_animals'

    plt.plot(np.linspace(np.min(ilds), np.max(ilds), n_points), psych_curve.fit, color=color, mfc=color, ls='-',
             label=label, alpha=alpha)

    # Move extreme datapoints closer to the center to zoom in
    psych_curve.xdata[0] = -20
    psych_curve.xdata[-1] = 20

    plt.errorbar(psych_curve.xdata, psych_curve.ydata, yerr=psych_curve.fit_error, color=color, fmt='o',
                 mfc=color)

    # plt.xscale('symlog', linthreshx=20)  # Set symmetric logarithmic spacing to zoom in the middle
    # plt.minorticks_off()  # Remove minor ticks
    plt.axhline(0.5, color='tab:gray', ls='--')
    plt.axvline(0., color='tab:gray', ls='--')
    plt.title(f'N={len(animals)}, {trials} trials')
    # plt.title(f'N={len(df.Setup.unique())}, {len(df)} trials')
    # plt.xlim([ilds[0] - 7, ilds[-1] + 7])
    plt.xlim([-21, 21])  # To chop the extreme values
    ilds[0] = -20
    ilds[-1] = 20
    plt.xticks(ilds)
    plt.gca().set_xticklabels(['-70', '-8', '', '', '0', '', '', '8', '70'])
    plt.ylim([-0.025, 1.025])
    plt.yticks([0, 0.5, 1])
    # plt.legend(loc='lower center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    sensitivity, bias, lr_lower, lr_upper = psych_curve.params
    plt.annotate("S=" + str(round(sensitivity, 2)) + "\n" +  # Sensitivity
                 "B=" + str(round(bias, 2)) + "\n" +  # Bias
                 lower_lapse + str(round(lr_lower, 2)) + "\n" +  # Upper lapse rate
                 upper_lapse + str(round(lr_upper, 2)),  # Lower lapse rate
                 xy=xy, xytext=xytext, color=color,
                 va=va, ha=ha, fontsize=fontsize)

    if save:
        # folder_out = '/home/alexis/Documentos/psychometric curves/'
        folder_out = Path.home() / 'Documentos/psychometric curves/'
        if not os.path.exists(folder_out):
            os.mkdir(folder_out)
        os.chdir(folder_out)
        plt.savefig(folder_out + filename, format=format, transparent=transparent)
        plt.close()

    return np.array(params)


########################################################################################################################

# # PROB. REP
# # 332
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220520-111627/332_stage_training_v2_20220520-111627.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220521-123227/332_stage_training_v2_20220521-123227.csv'
# pc_session_comparison(path1, path2, kind='prob_rep', color='tab:brown', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220523-111705/332_stage_training_v2_20220523-111705.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220524-111056/332_stage_training_v2_20220524-111056.csv'
# pc_session_comparison(path1, path2, kind='prob_rep', color='tab:brown', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220601-123831/332_stage_training_v2_20220601-123831.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220602-112855/332_stage_training_v2_20220602-112855.csv'
# pc_session_comparison(path1, path2, kind='prob_rep', color='tab:brown', save=True, format='png', transparent=False)
#
# # 333
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220515-112535/333_stage_training_v2_20220515-112535.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220516-113238/333_stage_training_v2_20220516-113238.csv'
# pc_session_comparison(path1, path2, kind='prob_rep', color='tab:brown', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220518-102053/333_stage_training_v2_20220518-102053.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220520-111416/333_stage_training_v2_20220520-111416.csv'
# pc_session_comparison(path1, path2, kind='prob_rep', color='tab:brown', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220523-111330/333_stage_training_v2_20220523-111330.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220524-110827/333_stage_training_v2_20220524-110827.csv'
# pc_session_comparison(path1, path2, kind='prob_rep', color='tab:brown', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220601-123634/333_stage_training_v2_20220601-123634.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220602-112547/333_stage_training_v2_20220602-112547.csv'
# pc_session_comparison(path1, path2, kind='prob_rep', color='tab:brown', save=True, format='png', transparent=False)
#
# # 337
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/337/sessions/337_stage_training_v2_20220519-105338/337_stage_training_v2_20220519-105338.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/337/sessions/337_stage_training_v2_20220520-112736/337_stage_training_v2_20220520-112736.csv'
# pc_session_comparison(path1, path2, kind='prob_rep', color='tab:brown', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/337/sessions/337_stage_training_v2_20220523-111832/337_stage_training_v2_20220523-111832.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/337/sessions/337_stage_training_v2_20220524-111333/337_stage_training_v2_20220524-111333.csv'
# pc_session_comparison(path1, path2, kind='prob_rep', color='tab:brown', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/337/sessions/337_stage_training_v2_20220531-103028/337_stage_training_v2_20220531-103028.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/337/sessions/337_stage_training_v2_20220601-125410/337_stage_training_v2_20220601-125410.csv'
# pc_session_comparison(path1, path2, kind='prob_rep', color='tab:brown', save=True, format='png', transparent=False)
#
#
# # PROB. RIGHT
# # 332
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220520-111627/332_stage_training_v2_20220520-111627.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220521-123227/332_stage_training_v2_20220521-123227.csv'
# pc_session_comparison(path1, path2, kind='prob_right', color='tab:orange', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220523-111705/332_stage_training_v2_20220523-111705.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220524-111056/332_stage_training_v2_20220524-111056.csv'
# pc_session_comparison(path1, path2, kind='prob_right', color='tab:orange', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220601-123831/332_stage_training_v2_20220601-123831.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/332/sessions/332_stage_training_v2_20220602-112855/332_stage_training_v2_20220602-112855.csv'
# pc_session_comparison(path1, path2, kind='prob_right', color='tab:orange', save=True, format='png', transparent=False)
#
# # 333
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220515-112535/333_stage_training_v2_20220515-112535.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220516-113238/333_stage_training_v2_20220516-113238.csv'
# pc_session_comparison(path1, path2, kind='prob_right', color='tab:orange', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220518-102053/333_stage_training_v2_20220518-102053.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220520-111416/333_stage_training_v2_20220520-111416.csv'
# pc_session_comparison(path1, path2, kind='prob_right', color='tab:orange', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220523-111330/333_stage_training_v2_20220523-111330.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220524-110827/333_stage_training_v2_20220524-110827.csv'
# pc_session_comparison(path1, path2, kind='prob_right', color='tab:orange', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220601-123634/333_stage_training_v2_20220601-123634.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/333/sessions/333_stage_training_v2_20220602-112547/333_stage_training_v2_20220602-112547.csv'
# pc_session_comparison(path1, path2, kind='prob_right', color='tab:orange', save=True, format='png', transparent=False)
#
# # 337
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/337/sessions/337_stage_training_v2_20220519-105338/337_stage_training_v2_20220519-105338.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/337/sessions/337_stage_training_v2_20220520-112736/337_stage_training_v2_20220520-112736.csv'
# pc_session_comparison(path1, path2, kind='prob_right', color='tab:orange', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/337/sessions/337_stage_training_v2_20220523-111832/337_stage_training_v2_20220523-111832.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/337/sessions/337_stage_training_v2_20220524-111333/337_stage_training_v2_20220524-111333.csv'
# pc_session_comparison(path1, path2, kind='prob_right', color='tab:orange', save=True, format='png', transparent=False)
# path1 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/337/sessions/337_stage_training_v2_20220531-103028/337_stage_training_v2_20220531-103028.csv'
# path2 = '/home/alexis/pv_nmdar_eranet/experiments/2AFC_2/setups/337/sessions/337_stage_training_v2_20220601-125410/337_stage_training_v2_20220601-125410.csv'
# pc_session_comparison(path1, path2, kind='prob_right', color='tab:orange', save=True, format='png', transparent=False)
