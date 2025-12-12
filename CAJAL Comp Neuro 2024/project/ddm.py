import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D

# Plotting parameters
sns.set_theme()
sns.set_style('white')
sns.set_style('ticks')
# sns.set_context('poster')


def classic_ddm(t_max=1, dt=0.001, v=1, w=0, a=1):
    """
    Simulate trials of a DDM
    :param t_max: Duration of the trial in seconds
    :param dt: Time step
    :param v: drift rate (or evidence strength, slope of the decision variable)
    :param w: starting point
    :param a: bound separation (distance between the two bounds)
    :return: x: decision variable, choice: choice made, rt: reaction time
    """

    # Think of how to initialize things
    time = np.arange(0, t_max, dt)  # Time vector
    mu = 0  # Mean of the noise
    sigma = np.sqrt(dt)  # Standard deviation of the noise
    x = w * a  # Initialize decision variable
    X = [x]  # Vector of all values of DV

    # Loop over time (while bound has not been hit)
    for t in range(len(time) - 1):
        noise = np.random.normal(mu, sigma)  # Draw a sample from the normal distribution. Diffusion term

        x += v * dt + noise  # Update the decision variable
        X.append(x)  # Append the new value to the vector of all values of DV

        # Check if the bound has been hit (trial ends)
        if x <= -a / 2:  # Lower bound
            # print(x)
            choice = -1  # Left choice
            X[-1] = -a / 2  # Set the value of the DV to the bound (so that it is not plotted beyond the bound
            break
        elif x >= a / 2:  # Upper bound
            # print(x)
            choice = 1  # Right choice
            X[-1] = a / 2  # Set the value of the DV to the bound (so that it is not plotted beyond the bound
            break
        else:  # No choice has been made yet (bound has not been hit)
            choice = -1 if x < w else 1

    rt = t * dt  # Reaction time
    return X, choice, rt


def reactive_ddm(t_max=1, t_fix=0.5, dt=0.001, v=0, w=0, a=1, sensory_delay=0, motor_delay=0):
    """
    Simulate trials of a DDM that include non-decision time
    :param t_max: Duration of the trial in seconds
    :param t_fix: fixation time
    :param dt: time step
    :param v: drift rate (or evidence strength, slope of the decision variable)
    :param w: starting point
    :param a: bound separation (distance between the two bounds)
    :param sensory_delay: sensory delay
    :param motor_delay: motor delay
    :return: x: decision variable, choice: choices made, rt: reaction time
    """

    # Think of how to initialize things
    time = np.arange(0, t_max, dt)  # Time vector
    mu = 0  # Mean of the noise
    sigma = np.sqrt(dt)  # Standard deviation of the noise
    x = w * a  # Initialize decision variable
    X = [x]  # Initialize the decision variable with the sensory delay

    # Loop over time (while bound has not been hit)
    for t in range(len(time) - 1):

        # Add sensory delay. Fixation time is also added here it is a different parameter
        if int((sensory_delay + t_fix) / dt) > t:
            x = w
            X.append(x)
        else:
            noise = np.random.normal(mu, sigma)  # Draw a sample from the normal distribution. Diffusion term
            x += v * dt + noise  # Update the decision variable
            X.append(x)  # Append the new value to the vector of all values of DV

        # Check if the bound has been hit (trial ends)
        if x <= -a / 2:  # Lower bound
            # print(x)
            choice = -1  # Left choice
            X[-1] = -a / 2  # Set the value of the DV to the bound (so that it is not plotted beyond the bound
            break
        elif x >= a / 2:  # Upper bound
            # print(x)
            choice = 1  # Right choice
            X[-1] = a / 2  # Set the value of the DV to the bound (so that it is not plotted beyond the bound
            break
        else:  # No choice has been made yet (bound has not been hit)
            choice = 0

    rt = t * dt  # Reaction time
    rt += motor_delay  # Add motor delay
    return X, choice, rt


def proactive_ddm(t_max=1, dt=0.001, v=0, w=0, a=1, motor_delay=0):
    """
    Simulate trials of a DDM that include non-decision time
    :param t_max: Duration of the trial in seconds
    :param dt: time step
    :param v: drift rate (or evidence strength, slope of the decision variable)
    :param w: starting point
    :param a: distance of single bound (opposed as separation between bounds as before)
    :param motor_delay: motor delay
    :return: x: decision variable, choices: choices made, RTs: reaction times
    """

    # Think of how to initialize things
    time = np.arange(0, t_max, dt)  # Time vector
    mu = 0  # Mean of the noise
    sigma = np.sqrt(dt)  # Standard deviation of the noise
    x = w * a  # Initialize decision variable
    X = [x]  # Vector of all values of DV

    # Loop over time (while bound has not been hit)
    for t in range(len(time) - 1):
        noise = np.random.normal(mu, sigma)  # Draw a sample from the normal distribution. Diffusion term

        x += v * dt + noise  # Update the decision variable
        X.append(x)  # Append the new value to the vector of all values of DV

        # Single bound (go signal)
        if x >= a:  # Upper bound
            # print(x)
            choice = 1  # Right choice
            X[-1] = a  # Set the value of the DV to the bound (so that it is not plotted beyond the bound
            break
        else:  # No choice has been made yet (bound has not been hit)
            choice = 0

    rt = t * dt + motor_delay  # Reaction time includes a fixed motor delay
    return X, choice, rt


def plot_reactive_ddm(N=1000, t_max=1, t_fix=0.5, dt=0.001, v=1, w=0, a=1, sensory_delay=0,
                      motor_delay=0):
    """
    Simulate and plot N trials of a DDM
    :param N: number of trials to simulate
    :param t_max: duration of the trial
    :param t_fix: fixation time
    :param dt: time step
    :param v: drift rate
    :param w: starting point
    :param a: bound separation
    :param sensory_delay: sensory delay
    :param motor_delay: motor delay
    """

    left_rts = []  # -1
    right_rts = []  # +1

    # plt.figure(constrained_layout=True)

    # Middle plot: DVs
    ax1 = plt.subplot(3, 1, 2)
    ax1.spines[['right', 'top', 'bottom']].set_visible(False)

    for i in range(N):

        X, choice, rt = reactive_ddm(t_max=t_max, t_fix=t_fix, dt=dt, v=v, w=w, a=a, sensory_delay=sensory_delay,
                                     motor_delay=motor_delay)

        if choice == -1:
            color = 'tab:blue'
            left_rts.append(rt)
        elif choice == 1:
            color = 'tab:orange'
            right_rts.append(rt)
        else:
            color = 'tab:gray'

        plt.plot(dt * np.arange(0, len(X)), X, color=color, alpha=0.1)
        # plt.plot(dt * np.arange(0, len(X)), X, color=color)

    # Plot DVs
    # plt.title(f'DDM simulation of {N} trials')
    # plt.xlabel('Time')
    plt.ylabel('DV')
    plt.yticks([-a / 2, 0, a / 2], [r'-$\theta$', '0', r'$\theta$'])  # Set 3 yticks only -a/2, 0, a/2
    plt.axhline(-a / 2, xmin=0, xmax=t_max, c='tab:blue', ls='--')
    plt.axhline(a / 2, xmin=0, xmax=t_max, c='tab:orange', ls='--')
    plt.axvline(t_fix, c='tab:gray', ls='--')
    plt.xticks([])
    ax1.set_xlim([0, t_max])

    left_handle = Line2D([], [], color='tab:blue', label='Left')
    right_handle = Line2D([], [], color='tab:orange', label='Right')
    ax1.legend(handles=[left_handle, right_handle], frameon=False, loc='upper right')

    # Plot Right RTs
    ax0 = plt.subplot(3, 1, 1)
    plt.hist(right_rts, bins=np.arange(0, t_max, dt * 20), color='tab:orange')
    plt.ylabel('Right RTs')
    # sns.despine(top=True, right=True, left=False, bottom=True)
    ax0.spines[['right', 'top', 'bottom']].set_visible(False)
    plt.xticks([])
    ax0.set_xlim([0, t_max])
    plt.axvline(t_fix, c='tab:gray', ls='--')

    # Plot Left RTs
    ax2 = plt.subplot(3, 1, 3)
    ax2.invert_yaxis()
    plt.hist(left_rts, bins=np.arange(0, t_max, dt * 20), color='tab:blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Left RTs')
    # sns.despine(top=True, right=True, left=False, bottom=True)
    ax2.spines[['right', 'top', 'bottom']].set_visible(False)
    # plt.xticks([])
    ax2.set_xlim([0, t_max])
    plt.axvline(t_fix, c='tab:gray', ls='--')

    # Set the same yaxis for the left and right RTs
    rts_ylims = [ax0.get_ylim(), ax2.get_ylim()]
    ax0.set_ylim([0, max([rts_ylims[0][1], rts_ylims[1][0]])])
    ax2.set_ylim([0, max([rts_ylims[0][1], rts_ylims[1][0]])])
    ax2.invert_yaxis()

    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.suptitle(f'Reactive DDM')


def plot_proactive_ddm(N=1000, t_max=1, dt=0.001, v=1, w=0, a=1, motor_delay=0):
    """
    Simulate and plot N trials of a DDM
    :param N: number of trials to simulate
    :param t_max: duration of the trial
    :param t_fix: fixation time
    :param dt: time step
    :param v: drift rate
    :param w: starting point
    :param a: bound separation
    :param motor_delay: motor delay
    """

    right_rts = []  # +1

    plt.figure(constrained_layout=True)

    # Middle plot: DVs
    ax1 = plt.subplot(2, 1, 2)
    ax1.spines[['right', 'top', 'bottom']].set_visible(False)

    for i in range(N):

        X, choice, rt = proactive_ddm(t_max=t_max, dt=dt, v=v, w=w, a=a, motor_delay=motor_delay)

        if choice == 1:
            color = 'tab:red'
            right_rts.append(rt)
        else:
            color = 'tab:gray'

        plt.plot(dt * np.arange(0, len(X)), X, color=color, alpha=0.1)

    # Plot DVs
    # plt.title(f'DDM simulation of {N} trials')
    plt.xlabel('Time')
    plt.ylabel('DV')
    plt.yticks([0, a], [r'0', r'$\theta$'])  # Set 2 yticks only 0, a/2
    plt.axhline(a, xmin=0, xmax=t_max, c='red', ls='--')
    # plt.xticks([])
    plt.axvline(0.5, c='tab:gray', ls='--')


    # Plot Right RTs
    ax0 = plt.subplot(2, 1, 1)
    plt.hist(right_rts, bins=np.arange(0, t_max, dt * 20), color='tab:red')
    plt.ylabel('RTs')
    # sns.despine(top=True, right=True, left=False, bottom=True)
    ax0.spines[['right', 'top', 'bottom']].set_visible(False)
    plt.xticks([])
    plt.axvline(0.5, c='tab:gray', ls='--')


    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(f'Proactive DDM')


def psiam_ddm(t_max=1.5, t_fix=0.5, dt=0.001, v=1, w=0, a=1, sensory_delay=0.02, motor_delay=0.08):
    """
    Simulate Parallel Sensory Integration and Action Model (PSIAM)
    From 'Proactive and reactive accumulation-to-bound processes compete during perceptual decisions'
    https://www.nature.com/articles/s41467-021-27302-8

    :param N: number of trials to simulate
    :param t_fixation: fixation time
    :param dt: time step
    :return: rts: reaction times, Xs: decision variables, choices: choices made, winner: 0: proactive, 1: reactive
    """

    # Run reactive and proactive trajectories
    X_re, choice_re, rt_re = reactive_ddm(t_max=t_max, t_fix=t_fix, dt=dt, v=v, w=w, a=a,
                                             sensory_delay=sensory_delay, motor_delay=motor_delay)
    X_pro, choice_pro, rt_pro = proactive_ddm(t_max=t_max, dt=dt, v=v, w=w, a=a, motor_delay=motor_delay)

    # Check which process won the race at every trial by comparing their RTs. 2 scenarios:
    # 1. Proactive won. Take the RT of the proactive process and read the choice from last value of the DV from the
    # reactive process at that time step
    # 2. Reactive won. Take the RT and choice of the reactive process (standard DDM)

    # 1. Proactive won
    if rt_pro < rt_re:
        # print('Proactive process won')
        winner = 0  # Proactive process won
        rt = rt_pro
        X = np.nan

        # If the proactive process won before the fixation time it is an abort trial, else it is a valid trial
        if rt_pro < t_fix:
            choice = 0
        else:
            # If the reactive process hit the bound during the motor delay of the proactive process, when trying to
            # read the DV at the time of the RT of the proactive process, it will be out of bounds. In this case,
            # read choice from last value of the DV of the reactive process (should be the same value as the bound)
            readout_time_step = int(rt_pro / dt)
            if readout_time_step >= len(X_re):
                dv = X_re[-1]
                choice = np.sign(dv)
            else:
                dv = X_re[int(rt_pro / dt)]
                choice = np.sign(dv)

    # 2. Reactive won
    else:
        # print('Reactive process won')
        winner = 1  # Reactive process won
        rt = rt_re
        X = X_re
        choice = choice_re

    return rt, X, choice, winner


def simulate_trajectories(N=1000, kind='psiam', t_max=1.5, t_fix=0.5, dt=0.001, v=1, w=0, a=1, sensory_delay=0.02,
                          motor_delay=0.08):

    Xs = []
    choices = []
    rts = []
    winners = []

    for i in range(N):
        if kind == 'classic':
            rt, X, choice = classic_ddm(t_max=t_max, dt=dt, v=v, w=w, a=a)
        elif kind == 'reactive':
            rt, X, choice = reactive_ddm(t_max=t_max, t_fix=t_fix, dt=dt, v=v, w=w, a=a,
                                         sensory_delay=sensory_delay, motor_delay=motor_delay)
        elif kind == 'proactive':
            rt, X, choice = proactive_ddm(t_max=t_max, dt=dt, v=v, w=w, a=a, motor_delay=motor_delay)
        elif kind == 'psiam':
            rt, X, choice, winner = psiam_ddm(t_max=t_max, t_fix=t_fix, dt=dt, v=v, w=w, a=a,
                                              sensory_delay=sensory_delay, motor_delay=motor_delay)
            winners.append(winner)

        Xs.append(X)
        choices.append(choice)
        rts.append(rt)

    # Store function inputs in a dictionary
    params = {'kind': kind, 'N': N, 't_max': t_max, 't_fix': t_fix, 'dt': dt, 'v': v, 'w': w, 'a': a,
              'sensory_delay': sensory_delay, 'motor_delay': motor_delay}

    return Xs, choices, rts, winners, params


def plot_rts(choices, rts, params):

    # Initialize lists
    correct_rts = []
    error_rts = []
    abort_rts = []

    for i in range(len(rts)):
        if choices[i] == 0:
            abort_rts.append(rts[i])
        elif choices[i] == np.sign(params['v']):
            correct_rts.append(rts[i])
        elif choices[i] != np.sign(params['v']):
            error_rts.append(rts[i])

    plt.figure(constrained_layout=True)

    bins = np.arange(0, params['t_max'], params['dt'] * 20)

    # Plot distribution of correct RTs
    ax1 = plt.subplot(2, 1, 1)
    ax1.hist(correct_rts, bins=bins, color='tab:green', label='correct')
    ax1.axvline(params['t_fix'], color='tab:gray', ls='--')
    ax1.xaxis.set_visible(False)
    sns.despine(top=True, right=True, left=False, bottom=True)

    # Plot distribution of error RTs
    ax2 = plt.subplot(2, 1, 2)
    ax2.hist(error_rts, bins=bins, color='tab:red', label='error')
    ax2.axvline(params['t_fix'], color='tab:gray', ls='--')
    sns.despine(top=True, right=True, left=False, bottom=True)

    # Plot distribution of abort RTs
    ax1.hist(abort_rts, bins=bins, color='tab:gray', label='abort')

    # Make both axes to have the same ylimits
    rts_ylims = [ax1.get_ylim(), ax2.get_ylim()]
    ax1.set_ylim([0, max([rts_ylims[0][1], rts_ylims[1][0]])])
    ax2.set_ylim([0, max([rts_ylims[0][1], rts_ylims[1][0]])])

    # Invert y-axis of the error plot
    ax2.invert_yaxis()

    # Remove vertical space between subplots
    plt.subplots_adjust(hspace=0)

    plt.xlabel('Time (s)')
    plt.suptitle(f'Correct vs error RTs')

    return bins, correct_rts, error_rts, abort_rts


def plot_rts_v2(choices, rts, params):

    # Initialize lists
    correct_rts = []
    error_rts = []
    abort_rts = []

    for i in range(len(rts)):
        if choices[i] == 0:
            abort_rts.append(rts[i])
        elif choices[i] == np.sign(params['v']):
            correct_rts.append(rts[i])
        elif choices[i] != np.sign(params['v']):
            error_rts.append(rts[i])

    plt.figure(constrained_layout=True)

    bins = np.arange(0, params['t_max'], params['dt'] * 20)

    # Plot distribution of correct RTs
    # ax1 = plt.subplot(2, 1, 1)
    plt.hist(correct_rts, bins=bins, color='tab:green', alpha=0.5, label='correct')
    plt.axvline(params['t_fix'], color='tab:gray', ls='--')
    # plt.xaxis.set_visible(False)
    sns.despine(top=True, right=True, left=False, bottom=True)

    # Plot distribution of error RTs
    # ax2 = plt.subplot(2, 1, 2)
    plt.hist(error_rts, bins=bins, color='tab:red', alpha=0.5, label='error')
    # plt.axvline(params['t_fix'], color='tab:gray', ls='--')
    sns.despine()

    # Plot distribution of abort RTs
    plt.hist(abort_rts, bins=bins, color='tab:gray', alpha=0.5, label='abort')

    # Make both axes to have the same ylimits
    # rts_ylims = [ax1.get_ylim(), ax2.get_ylim()]
    # ax1.set_ylim([0, max([rts_ylims[0][1], rts_ylims[1][0]])])
    # ax2.set_ylim([0, max([rts_ylims[0][1], rts_ylims[1][0]])])

    # Invert y-axis of the error plot
    # ax2.invert_yaxis()

    # Remove vertical space between subplots
    # plt.subplots_adjust(hspace=0)

    plt.xlabel('Time (s)')
    plt.suptitle(f'Correct vs error RTs')
    plt.legend(frameon=False)

    return bins, correct_rts, error_rts, abort_rts


def plot_tachometric(choices, rts, params):

    # Initialize lists
    correct_rts = []
    error_rts = []
    abort_rts = []

    for i in range(len(rts)):

        if choices[i] == 0:
            abort_rts.append(rts[i])
            # Assign random choices for abort trials
            coin_flip = np.random.choice([-1, 1])
            if coin_flip == np.sign(params['v']):
                correct_rts.append(rts[i])
            else:
                error_rts.append(rts[i])
        elif choices[i] == np.sign(params['v']):
            correct_rts.append(rts[i])
        elif choices[i] != np.sign(params['v']):
            error_rts.append(rts[i])

    bins = np.arange(0, params['t_max'], params['dt'] * 20)

    hist_correct, bin_edges_correct = np.histogram(correct_rts, bins=bins)
    hist_error, bin_edges_error = np.histogram(error_rts, bins=bins)
    hist = hist_correct / (hist_correct + hist_error)
    bins_centers = (bin_edges_correct[1:] + bin_edges_correct[:-1]) / 2

    # Plot the tachometric curve
    plt.figure(constrained_layout=True)
    plt.plot(bins_centers, hist)
    plt.xlabel('RTs (s)')
    plt.ylabel('Accuracy')
    plt.title(f'Tachometric curve')
    # plt.xlim([0.5, t_max])
    xticks = np.arange(params['t_fix'], params['t_max'] + params['t_fix'], params['t_fix'])
    plt.xticks(xticks)
    sns.despine()

    return bins_centers, hist


def plot_responses(winners, rts, params):

    # Initialize lists
    proactive_rts = []  # 0
    reactive_rts = []  # 1

    for i in range(len(rts)):

        if winners[i] == 0:
            proactive_rts.append(rts[i])
        elif winners[i] == 1:
            reactive_rts.append(rts[i])

    bins = np.arange(0, params['t_max'], params['dt'] * 20)

    hist_reactive, bin_edges_reactive = np.histogram(reactive_rts, bins=bins)
    hist_proactive, bin_edges_proactive = np.histogram(proactive_rts, bins=bins)

    # Normalize
    hist_total = hist_reactive + hist_proactive
    hist_reactive = hist_reactive / hist_total
    hist_proactive = hist_proactive / hist_total
    bins_centers = (bin_edges_reactive[1:] + bin_edges_proactive[:-1]) / 2

    # Plot the tachometric curve
    plt.figure(constrained_layout=True)
    plt.plot(bins_centers, hist_reactive, color='tab:green', label='Reactive')
    plt.plot(bins_centers, hist_proactive, color='tab:red', label='Proactive')
    plt.xlabel('RTs (s)')
    plt.ylabel('Proportion of responses')
    # plt.xlim([0.5, t_max])
    xticks = np.arange(params['t_fix'], params['t_max'] + params['t_fix'], params['t_fix'])
    plt.xticks(xticks)
    plt.legend(frameon=False)
    sns.despine()

    return bins_centers, hist_reactive, hist_proactive

"""
N=10000
kind='psiam'
t_max=1.5
t_fix=0.5
dt=0.001
v=1
w=0
a=1
sensory_delay=0.02
motor_delay=0.8


# RTs
bins_RTs = []
correct_RTs = []
error_RTs = []
abort_RTs = []

# Tachometric curve
bins_tach = []
hists_tach = []

# Responses
bins_responses = []
hist_reactive_responses = []
hist_proactive_responses = []


delays = [0.02, 0.08]
inverted_delays = [0.08, 0.02]

delays = np.arange(0, 0.11, 0.01)
inverted_delays = delays[::-1]


for i in range(len(delays)):
    print(delays[i], inverted_delays[i])
    Xs, choices, rts, winners, params = simulate_trajectories(N=N, kind=kind, t_max=t_max, t_fix=t_fix, dt=dt, v=v, w=w,
                                                              a=a, sensory_delay=delays[i], motor_delay=inverted_delays[i])

    # RTs
    bins, correct_rts, error_rts, abort_rts = plot_rts_v2(choices, rts, params)
    bins_RTs.append(bins)
    correct_RTs.append(correct_rts)
    error_RTs.append(error_rts)
    abort_RTs.append(abort_rts)

    # Tachometric curve
    bins_centers, hist = plot_tachometric(choices, rts, params)
    bins_tach.append(bins_centers)
    hists_tach.append(hist)

    # Responses
    bins_centers, hist_reactive, hist_proactive = plot_responses(winners, rts, params)
    bins_responses.append(bins_centers)
    hist_reactive_responses.append(hist_reactive)
    hist_proactive_responses.append(hist_proactive)


# RTs
plt.figure(constrained_layout=True)
for i in range(len(delays)):
    plt.hist(correct_RTs[i] + error_RTs[i] + abort_RTs[i], bins, alpha=0.5,  label=f'$S_D=${delays[i]}, $M_D={inverted_delays[i]}$')
    # plt.hist(correct_RTs[i] + error_RTs[i] + abort_rts[i], bins, alpha=0.5, label=f'$S_D=${delays[i]}, $M_D={inverted_delays[i]}$')

plt.xlabel('Time (s)')
# plt.ylabel('Accuracy')
# plt.title(f'Tachometric curve')
plt.legend(frameon=False)
plt.show()
sns.despine()
plt.axvline(t_fix, c='tab:gray', ls='--')

colors = ['tab:blue', 'tab:blue','tab:blue','tab:blue','tab:blue', 'tab:gray', 'tab:orange', 'tab:orange', 'tab:orange', 'tab:orange', 'tab:orange',]
alpha = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# Tachometric curve
plt.figure(constrained_layout=True)
for i in range(len(delays)):
    print(i)
    if i ==0:
        label=f'$S_D<S_M$'
    elif i ==10:
        label=f'$S_D>S_M$'
    else:
        label=''
    plt.plot(bins_tach[i], hists_tach[i], color=colors[i], alpha=alpha[i], label=label)
    # plt.plot(bins_tach[i], hists_tach[i], color=colors[i], alpha=alpha[i], label=f'$S_D=${delays[i]}, $M_D={inverted_delays[i]}$')

plt.xlabel('RTs (s)')
plt.ylabel('Accuracy')
plt.title(f'Tachometric curve ($N=${N} trials)')
plt.legend(frameon=False)
plt.ylim([0, 1])
#plt.xlim([0.45, 0.75])
plt.show()
sns.despine()
plt.axvline(t_fix, c='tab:gray', ls='--')

# Make color gradient from clear to dark in orange with 5 colors:


# Responses
plt.figure(constrained_layout=True)
alpha = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
greens=['green', 'darkgreen']
reds=['red', 'darkred']
for i in range(len(delays)):
    # Only label the first plot
    if i == 0:
        plt.plot(bins_responses[i], hist_proactive_responses[i], color='tab:green', label='Reactive')
        plt.plot(bins_responses[i], hist_reactive_responses[i], color='tab:red', label='Proactive')
    else:

        plt.plot(bins_responses[i], hist_proactive_responses[i], color='tab:green', label='')
        plt.plot(bins_responses[i], hist_reactive_responses[i], color='tab:red', label='')

plt.xlabel('RTs (s)')
plt.ylabel('Proportion of responses')
plt.legend(frameon=False)
plt.show()
sns.despine()
plt.axvline(t_fix, c='tab:gray', ls='--')









# Xs, choices, rts, winners, params = simulate_trajectories(N=N, kind=kind, t_max=t_max, t_fix=t_fix, dt=dt, v=v, w=w,
#                                                           a=a, sensory_delay=sensory_delay, motor_delay=motor_delay)

# bins, correct_rts, error_rts, abort_rts = plot_rts(choices, rts, params)
# bins_centers, hist = plot_tachometric(choices, rts, params)
# bins_centers, hist_reactive, hist_proactive = plot_responses(winners, rts, params)

# Xs, choices, rts, winners, params = simulate_trajectories(N=10000, kind='psiam', t_max=1.5, t_fix=0.5, dt=0.001, v=1,
#                                                          w=0, a=1, sensory_delay=0.02, motor_delay=0.08)


########################################################################################################################

# Make a pandas Dataframe from the lists the Xs, choices, rts, winners, params
# df = pd.DataFrame({'choices': choices, 'rts': rts, 'winners': winners})
# df.to_csv('data.csv', index=False)

"""