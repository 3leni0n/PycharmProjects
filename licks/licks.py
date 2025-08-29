import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
import seaborn as sns


# Define functions to get licks and RTs
def curate_licks(licks: pd.Series, df_behavior: pd.DataFrame, time_window=1) -> pd.Series:
    """
    Curate licks for a given behavioral session (remove licks before Response Window opens or after ITI ends).
    Note that timestamps (in seconds) of licks and events are relative to the trial onset (0 s).
    :param licks: Series with a list of licks per trial
    :param df_behavior: DataFrame with the behavioral data of a session
    :param time_window: Time window to consider for licks (in seconds). This will result in N licks = lick rate (Hz)
    :return: Curated Series with l
    :return: Series with curated licks
    """

    premature_lick_trials = []  # List of indices of trials where licks happened before response window
    tolerance = 0.0  # In seconds, tolerance for licks outside the response window (delay of motor Arduino code)

    for trial in range(len(licks)):
        resp_win_start = df_behavior.RespWinStart.iloc[trial]
        resp_win_end = df_behavior.RespWinEnd.iloc[trial]

        # Check if any licks before response window start to detect trials in which the motor was stuck
        # (do not include in the condition licks after response window end, as these depend on the time the lickport was
        # available (which varies depending on trial outcome)
        if any((lick < resp_win_start - tolerance) for lick in licks.iloc[trial]):
            premature_lick_trials.append(trial)

        licks[trial] = [lick for lick in licks.iloc[trial]
                        if resp_win_start - tolerance <= lick <= resp_win_end + time_window + tolerance]

    return licks, premature_lick_trials


def get_peri_stim_licks(df_behavior, event='StimStart', time_window=1):
    """
    Get peri-stimulus licks for a given behavioral session
    :param df_behavior: DataFrame with behavior data
    :return: pd.Series with a list of peri-stimulus licks per trial
    """

    # Convert string representations of lists back to actual lists
    try:
        df_behavior["Port1In"] = df_behavior["Port1In"].apply(ast.literal_eval)
        df_behavior["Port2In"] = df_behavior["Port2In"].apply(ast.literal_eval)
    except ValueError:
        print('Port1In or Port2In are already lists')

    licks_left = df_behavior.Port1In.copy()
    licks_right = df_behavior.Port2In.copy()

    # Curate licks
    licks_left, premature_lick_trials_left = curate_licks(licks_left, df_behavior, time_window)
    licks_right, premature_lick_trials_right = curate_licks(licks_right, df_behavior, time_window)

    for trial in range(len(df_behavior)):

        # Align licks to the event time
        event_time = df_behavior[event].iloc[trial]
        licks_left[trial] = [x - event_time for x in licks_left.iloc[trial]]  # Left
        licks_right[trial] = [x - event_time for x in licks_right.iloc[trial]]  # Right

    licks = [licks_left] + [licks_right]
    premature_lick_trials = [premature_lick_trials_left] + [premature_lick_trials_right]

    return licks, premature_lick_trials


def compute_psth(peri_stim_licks, time_win=[-1, 3], bin_size=0.1):
    """
    Compute a PSTH of a given cluster aligned to a specific event.
    :param peri_stim_spikes: Lick times of a given subject (output of get_peri_stim_licks)
    :param time_win: List with the time window around the event (default: [-1, 3])
    :param bin_size: Size of the bins for the PSTH (default: 0.1 s)
    """

    n_bins = int((time_win[1] - time_win[0]) / bin_size) + 1
    bins = np.linspace(time_win[0], time_win[1], n_bins)  # linspace is preferred over arange for PSTHs

    psth = []
    # Loop over trials (timestamps of stimulus onset)
    for trial in range(len(peri_stim_licks)):
        hist, _ = np.histogram(peri_stim_licks.iloc[trial], bins)  # Ignore the bin_edges output
        psth.append(hist)
    psth = np.array(psth)  # Convert to numpy array

    return bins, psth


def inter_lick_interval(licks: pd.Series) -> list:
    """
    Compute the mean inter-lick interval (ILI) of the licks of a behavioral session.
    :return: Inter-lick interval (ILI) of the licks per trial
    """

    ili = []

    for trial in range(len(licks)):
        licks_trial = (licks.iloc[trial])
        licks_trial.sort()  # Sort licks in ascending order
        n_licks = len(licks_trial)
        if n_licks < 2:  # Minimum of 2 licks required to compute ILI
            ili.append(np.nan)
        else:
            intervals = np.diff(licks_trial)
            # ili.append(intervals[0])  # First ili (interval between 1st-2nd lick)
            ili.append(np.mean(intervals))  # Mean ili

    return ili


def get_rt(df_behavior):
    """
    Compute the reaction time (RT) of a behavioral session from EVENT data.
    :param df_behavior: DataFrame with the behavioral data
    :return: Reaction time (RT) of the licks per trial
    """

    rt = []

    for trial in range(len(df_behavior)):

        if df_behavior.Miss.iloc[trial] == 1:
            rt.append(np.nan)
        else:
            rt.append(df_behavior.RespWinLen.iloc[trial])

    return rt


def get_rt2(licks, df_behavior):
    """
    Compute the reaction time (RT) of a behavioral session from LICK data.
    :param df_behavior: DataFrame with the behavioral data
    :return: Reaction time (RT) of the licks per trial
    """

    rt2 = []

    # Combine left and right licks into a single Series
    licks = pd.Series([left + right for left, right in zip(licks[0], licks[1])])

    for trial in range(len(licks)):

        trial_licks = licks.iloc[trial]
        trial_licks.sort()  # Sort licks in ascending order

        if not trial_licks or df_behavior.Miss.iloc[trial] == 1:  # If no licks in the trial
            rt2.append(np.nan)
        else:
            # Align response window start to stimulus onset
            stim_start = df_behavior.StimStart.iloc[trial]
            resp_win_start = df_behavior.RespWinStart.iloc[trial]
            resp_win_start_aligned = resp_win_start - stim_start

            # Find the first lick within the response window
            first_lick = min(trial_licks)
            first_lick_aligned = first_lick - resp_win_start_aligned

            if first_lick_aligned < 0:  # If the first lick is before the response window
                rt2.append(np.nan)
            else:
                rt2.append(first_lick_aligned)  # Use the first lick as RT

    return rt2


def add_lick_data(df_behavior: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to compute licks and reaction times for a given behavioral session.
    :param df_behavior: DataFrame with the behavioral data of a session
    :return: DataFrame with additional columns for licks and reaction times
    """

    # Apply lick functions
    licks, premature_lick_trials = get_peri_stim_licks(df_behavior, event='StimStart')

    # Combine left and right premature lick trials
    premature_lick_trials = sorted(set(premature_lick_trials[0] + premature_lick_trials[1]))

    # Print % premature lick trials
    try:
        print(f'{len(premature_lick_trials) / len(df_behavior) * 100:.2f}% of premature lick trials (before response window)')
    except ZeroDivisionError:
        print('0% of premature lick trials (before response window)')

    # # Drop premature lick trials from DataFrame
    # df_behavior = df_behavior.drop(index=premature_lick_trials).reset_index(drop=True)

    licks_left = licks[0]
    licks_right = licks[1]
    n_licks_left = [len(lick) for lick in licks_left]
    n_licks_right = [len(lick) for lick in licks_right]
    # n_licks = np.where(df_behavior.Side == 0, n_licks_left, n_licks_right)  # N licks in correct side
    # n_licks = np.where(df_behavior.Choice == 0, n_licks_left, n_licks_right)  # N licks in chosen side
    n_licks = np.array(n_licks_left) + np.array(n_licks_right)  # Total N licks in both sides
    ili_left = inter_lick_interval(licks_left)
    ili_right = inter_lick_interval(licks_right)
    # ili = np.where(df_behavior.Side == 0, ili_left, ili_right)  # ILI in correct side
    # ili = np.where(df_behavior.Choice == 0, ili_left, ili_right)  # ILI in chosen side
    ili = [np.nanmean([l, r]) if not (np.isnan(l) and np.isnan(r)) else np.nan for l, r in zip(ili_left, ili_right)]
    rt = get_rt(df_behavior)

    # Add lick vars to DataFrame
    df_behavior['LicksLeft'] = licks[0]
    df_behavior['LicksRight'] = licks[1]
    df_behavior['nLicksLeft'] = n_licks_left
    df_behavior['nLicksRight'] = n_licks_right
    df_behavior['nLicks'] = n_licks
    df_behavior['leftILI'] = ili_left
    df_behavior['rightILI'] = ili_right
    df_behavior['ILI'] = ili
    df_behavior['RT'] = rt

    # Add RT2 to DataFrame
    licks, premature_lick_trials = get_peri_stim_licks(df_behavior, event='StimStart', time_window=0)
    rt2 = get_rt2(licks, df_behavior)
    df_behavior['RT2'] = rt2

    return df_behavior


########################################################################################################################

# Plotting functions

def plot_licks_dist(df_behavior, var='RT', density=False):
    """
    Plot the licks distribution for a variable of interest.
    :param df_behavior: DataFrame with the behavioral data
    :param var: Variable to plot (e.g., 'RT', 'nLicks', 'ILI')
    :param density: If True, plot density instead of frequency (default: False)
    :return: None
    """

    # plt.figure(constrained_layout=True)
    color = 'k'

    # Continuous variables
    if var == 'RT' or var == 'ILI':
        if density:
            ylabel = 'Density'
            sns.kdeplot(df_behavior[var], color=color)
        else:
            ylabel = 'Frequency'
            plt.hist(df_behavior[var], bins=1000, color=color, edgecolor=color)
        plt.xlim(0, 0.5)
        plt.xlabel('Time (s)')

    # Discrete variable
    elif var == 'nLicks':
        min_val = df_behavior[var].min()
        max_val = df_behavior[var].max()
        bins = np.arange(min_val - 0.5, max_val + 1.5, 1)  # Centers bins on integers
        plt.hist(df_behavior[var], bins=bins, color='black', density=True)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatic integer ticks
        plt.xlim(0, 20)
        # plt.xlim(min_val - 0.5, max_val + 0.5)
        plt.xlabel(var)
        if density:
            ylabel = 'Density'
        else:
            ylabel = 'Frequency'

    if df_behavior.Subject.unique().size > 1:
        title = (f'{var}\n'
                 f'N={len(df_behavior.Subject.unique())}, {len(df_behavior)/1000:.1f}k trials')
    else:
        title = (f'{var}\n'
                 f'{df_behavior.Subject.unique()[0]}, {len(df_behavior)/1000:.1f}k trials')

    plt.title(title)
    plt.ylabel(ylabel)
    sns.despine()


def plot_licks_split(df_behavior, var='RT', split='outcome', kind='kde'):
    """
    Plot the licks distribution of a variable of interest split by condition.
    :param df_behavior: DataFrame with the behavioral data
    :param var: Variable to plot (e.g., 'RT', 'nLicks', 'ILI')
    :param split: Split by 'outcome', 'choice', 'stim', 'rep_choice', 'rep_trial', 'prev_out', or 'session_half'
    :param kind: Kind of plot to use ('hist' or 'kde')
    :return:
    """

    # Split
    if split == 'outcome':
        split_var_name = 'Hit'
        colors = ['tab:red', 'tab:green']
        labels = ['Error', 'Hit']
    elif split == 'choice':
        split_var_name = 'Choice'
        colors = ['tab:blue', 'tab:orange']
        labels = ['Left', 'Right']
    elif split == 'stim':
        split_var_name = 'Side'
        colors = ['tab:blue', 'tab:orange']
        labels = ['Left', 'Right']
    elif split == 'rep_choice':
        split_var_name = 'RepChoice'
        colors = ['tab:purple', 'tab:brown']
        labels = ['Alt.', 'Rep.']
    elif split == 'rep_trial':
        split_var_name = 'RepTrial'
        colors = ['tab:purple', 'tab:brown']
        labels = ['Alt.', 'Rep.']
    elif split == 'prev_out':
        split_var_name = 'AfterHit'
        colors = ['tab:red', 'tab:green']
        labels = ['Error', 'Hit']
    elif split == 'half':
        split_var_name = 'SessionHalf'
        colors = ['tab:blue', 'tab:orange']
        labels = ['1st half', '2nd half']
    elif split == 'drug':
        split_var_name = 'Drug'
        colors = ['tab:gray', 'tab:pink']
        labels = ['Saline', 'Drug']

    # plt.figure(constrained_layout=True)

    for i in range(2):

        split_var = df_behavior[df_behavior[split_var_name] == i][var]

        # Continuous variables
        if var == 'RT' or var == 'ILI':
            if kind == 'hist':
                plt.hist(split_var, bins=1000, density=False, label=labels[i], color=colors[i],
                         edgecolor='none', alpha=0.5)
                ylabel = 'Frequency'
            elif kind == 'kde':
                sns.kdeplot(split_var, color=colors[i], label=labels[i])
                ylabel = 'Density'
            plt.xlim(0, 0.5)
            plt.xlabel('Time (s)')

        # Discrete variable
        elif var == 'nLicks':
            min_val = split_var.min()
            max_val = split_var.max()
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)  # Centers bins on integers
            density = True if kind == 'kde' else False
            plt.hist(split_var, bins=bins, density=density, histtype='step', color=colors[i], label=labels[i])
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatic integer ticks
            plt.xlim(0, 20)
            # plt.xlim(min_val - 0.5, max_val + 0.5)
            plt.xlabel(var)
            if kind == 'kde':
                ylabel = 'Density'
            else:
                ylabel = 'Frequency'

        if df_behavior.Subject.unique().size > 1:
            title = (f'{var}\n'
                     f'N={len(df_behavior.Subject.unique())}, {len(df_behavior)/1000:.1f}k trials')
        else:
            title = (f'{var}\n'
                     f'{df_behavior.Subject.unique()[0]}, {len(df_behavior)/1000:.1f}k trials')

    plt.legend(frameon=False)
    plt.title(title)
    plt.ylabel(ylabel)
    sns.despine()


def plot_ild_dist(df_behavior, var='RT', insets=True):
    """
    Plot the licks distribution of a variable of interest split by absolute ILD levels.
    :param df_behavior: DataFrame with the behavioral data of a session
    :param var: Reaction Time (RT) or number of licks (nLicks)
    :return:
    """

    # plt.figure(constrained_layout=True)

    # Collapse the signed ILD levels to absolute values for cleaner visualization
    abs_ilds = sorted(df_behavior.absILD.unique().astype(int), reverse=True)
    palette = list(sns.color_palette('tab10', len(abs_ilds)))[::-1]

    peaks = {}
    # Plot the distribution for each absolute ILD level
    for i, ild in enumerate(abs_ilds):
        df_ild = df_behavior[df_behavior.absILD == ild]
        color = palette[i]

        # Continuous variables
        if var == 'RT' or var == 'ILI':
            # Plot and capture the Line2D object
            sns.kdeplot(df_ild[var], color=color, label=ild)
            # sns.histplot(df_ild[var], stat='density', element='step', fill=False, kde=False, color=color, label=ild)
            plt.xlim(0, 0.5)
            loc = 'upper center'

            # Extract x and y data from the plotted line
            line = plt.gca().lines[-1]
            x, y = line.get_data()

            # Find the peak (x at maximum y)
            peak_rt = x[np.argmax(y)]
            peaks[ild] = peak_rt
            print(f'ILD {ild}: peak {var} = {peak_rt:.3f} s')

        # Discrete variable
        elif var == 'nLicks':
            min_val = df_ild[var].min()
            max_val = df_ild[var].max()
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)  # Centers bins on integers
            plt.hist(df_ild[var], bins=bins, density=True, histtype='step', linewidth=1.5, color=color, label=ild)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatic integer ticks
            plt.xlim(0, 20)
            # plt.xlim(min_val - 0.5, max_val + 0.5)
            plt.xlabel(var)

            # nlicks = df_ild[var]
            # nlicks_mean = nlicks.mean()
            # nlicks_sem = sem(nlicks)
            # plt.errorbar(i, nlicks_mean, nlicks_sem, fmt='o', color=color, label=ild)
            loc = 'upper right'

    # # Print peaks
    # for ild, peak in peaks.items():
    #     print(f'ILD {ild}: peak {var} = {peak:.3f} s')

    mean_rt = np.mean(list((peaks.values())))
    print(f'Mean {var} = {mean_rt:.3f} s')

    plt.title(var + ' distribution')
    plt.xlabel(var)
    plt.ylabel('Density')
    plt.legend(loc=loc, frameon=False, title='|ILD|')
    sns.despine()

    if var == 'RT' and insets:

        # Zoomed inset on the peak of the distribution
        ax = plt.gca()  # get current axes
        ax_inset = inset_axes(ax, width='30%', height='30%', loc='upper right')
        xlim = (mean_rt - 0.025, mean_rt + 0.025)

        for i, line in enumerate(ax.lines):
            x = line.get_xdata()
            y = line.get_ydata()
            mask = (x >= xlim[0]) & (x <= xlim[1])
            color = palette[i]
            ax_inset.plot(x[mask], y[mask], label=line.get_label(), color=color)

        ax_inset.set_xlim(xlim)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        sns.despine(ax=ax_inset)

        # Zoomed inset on the second peak of the distribution
        ax_inset = inset_axes(ax, width='30%', height='30%', loc='center right')
        xlim = (0.15, 0.2)

        for i, line in enumerate(ax.lines):
            x = line.get_xdata()
            y = line.get_ydata()
            mask = (x >= xlim[0]) & (x <= xlim[1])
            color = palette[i]
            ax_inset.plot(x[mask], y[mask], label=line.get_label(), color=color)

        ax_inset.set_xlim(xlim)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        sns.despine(ax=ax_inset)


def plot_ild_dist_mean(df_behavior, var='RT'):
    """
    Plot the mean ± SEM of a variable for each absolute ILD level as a categorical bar plot.
    """
    abs_ilds = sorted(df_behavior.absILD.unique().astype(int))
    palette = list(sns.color_palette('tab10', len(abs_ilds)))

    means = []
    sems = []

    # Compute means and SEMs
    for ild in abs_ilds:
        df_ild = df_behavior[df_behavior.absILD == ild][var]
        mean_ild = df_ild.mean()
        sem_ild = df_ild.sem()
        means.append(mean_ild)
        sems.append(sem_ild)

    y_min = min(means) - 0.25 * (max(means) - min(means))
    x = np.arange(len(abs_ilds))  # positions for the bars
    # plt.bar(x, means, yerr=sems, color=palette)
    plt.bar(x, means - y_min, bottom=y_min, yerr=sems, color=palette)
    # plt.errorbar(x, means, sems, fmt='-o', color=color, label=label)

    plt.ylim(y_min, None)
    plt.xticks(x, abs_ilds)  # ILD values as category labels
    plt.xlabel('|ILD|')
    plt.ylabel(var)
    plt.title(f'Mean {var}')
    sns.despine()


def plot_licks_per_subject(df_behavior, plot_func, format='A4', **kwargs):
    """
    Plot a subplot per subject of a given plotting function. Adjust automatically the figure grid to fit all subjects.
    :param df_behavior: DataFrame containing the behavioral data.
    :param plot_func: Plotting function to be applied.
    :param kwargs: Keyword arguments to be passed to `plot_func`.
    :return: None
    """

    subjects = df_behavior['Subject'].unique()
    n_subj = len(subjects)

    # Layout: N columns, enough rows
    ncols = 3
    nrows = -(-n_subj // ncols)  # Ceiling division

    # Paper size (format)
    if format == 'A4':
        figsize = (8.27, 11.69)
    if format == 'A3':
        figsize = (11.69, 16.54)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True, sharey=True)

    for ax, subject in zip(axes.flatten(), subjects):
        df_subject = df_behavior[df_behavior['Subject'] == subject]
        plt.sca(ax)  # Make ax current
        plot_func(df_subject, **kwargs)
        title = (f'{df_behavior.Subject.unique()[0]}, {len(df_behavior) / 1000:.1f}k trials')
        ax.set_title(title)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # remove unused axes if any
    for ax in axes.flatten()[len(subjects):]:
        ax.remove()

    # fig.suptitle(var)
    fig.tight_layout()


def plot_chrono_curve(df_behavior, absolute=True):
    """
    Plot the chronometric curve of a behavioral session (all trials).
    :param df_behavior: DataFrame with the behavioral data of a session
    :param absolute: If True, plot the absolute value of ILD (default: True)
    :return:
    """

    if absolute:
        df_behavior['absILD'] = df_behavior['ILD'].abs()
        mean_rts = df_behavior.groupby('absILD')['RT'].mean().reset_index()
        ilds = sorted(df_behavior['absILD'].unique())
        x = mean_rts['absILD']
        xlabel = '|ILD|'
    else:
        mean_rts = df_behavior.groupby('ILD')['RT'].mean().reset_index()
        ilds = sorted(df_behavior['ILD'].unique())
        x = mean_rts['ILD']
        xlabel = 'ILD'

    if df_behavior.Subject.unique().size > 1:
        title = f'Chronometric Curve\n N={len(df_behavior.Subject.unique())} mice, {len(df_behavior)} trials'
    else:
        title = f'Chronometric Curve\n ID: {df_behavior.Subject.unique()[0]}, N={len(df_behavior)} trials'

    print(mean_rts)
    plt.figure(constrained_layout=True)
    y = mean_rts['RT']
    plt.plot(x, y, color='k', marker='o', linestyle='-')
    plt.xticks(ilds)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Mean RT (s)')
    plt.grid()
    sns.despine()


def plot_chrono_curve_split(df_behavior, split='outcome', absolute=True):
    """
    Plot the chronometric curve of a behavioral session split by outcome.
    :param df_behavior: DataFrame with the behavioral data of a session
    :param split: Split by 'outcome' or 'hit_error'
    :param absolute: If True, plot the absolute value of ILD (default: True)
    :return:
    """

    # Split
    if split == 'outcome':
        var = 'Hit'
        colors = ['tab:red', 'tab:green']
        labels = ['Error', 'Hit']
    elif split == 'choice':
        var = 'Choice'
        colors = ['tab:blue', 'tab:orange']
        labels = ['Left', 'Right']
    elif split == 'stim':
        var = 'Side'
        colors = ['tab:blue', 'tab:orange']
        labels = ['Left', 'Right']
    elif split == 'repeat':
        var = 'RepTrial'
        colors = ['tab:purple', 'tab:brown']
        labels = ['Alt.', 'Rep.']
    elif split == 'prev_out':
        var = 'AfterHit'
        colors = ['tab:red', 'tab:green']
        labels = ['Error', 'Hit']

    # Collapse trials by ILD or not
    if absolute:
        df_behavior['absILD'] = df_behavior['ILD'].abs()
        ilds = sorted(df_behavior['absILD'].unique())
        ild_col = 'absILD'
        xlabel = '|ILD|'
    else:
        ilds = sorted(df_behavior['ILD'].unique())
        ild_col = 'ILD'
        xlabel = 'ILD'

    plt.figure(constrained_layout=True)
    for i in range(2):

        subset = df_behavior[df_behavior[var] == i]
        mean_rts = subset.groupby(ild_col)['RT'].mean().reset_index()
        x = mean_rts[ild_col]
        y = mean_rts['RT']

        plt.plot(x, y, color=colors[i], marker='o', linestyle='-', label=labels[i])

    # Title
    if df_behavior.Subject.unique().size > 1:
        title = f'Chronometric Curve\n N={len(df_behavior.Subject.unique())} mice, {len(df_behavior)} trials'
    else:
        title = f'Chronometric Curve\n ID: {df_behavior.Subject.unique()[0]}, N={len(df_behavior)} trials'

    plt.xticks(ilds)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Mean RT (s)')
    plt.legend(frameon=False)
    plt.grid()
    sns.despine()
