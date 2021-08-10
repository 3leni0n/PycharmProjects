# Import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from my_fun.my_fun import ild, my_select_evidence, compute_psych_curve

df = ild()
df_summary = df.groupby('Evidence', as_index=False).mean()  # SQL-style index
n_trials = 1000
trial_types = [0, 1]  # 0=left, 1=right
trial_list = np.random.choice(trial_types, n_trials).tolist()  # Generate random trial vector of length n_trials
evidences = df.Evidence.unique()
target_evidences = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
target_ilds = list(df_summary.Mean[df_summary.Evidence.isin(target_evidences)].round())
ilds = df_summary.Mean[df_summary.Evidence.isin(evidences)].reset_index(drop=True, inplace=False).round()
df_sample = df[df.Evidence.isin(evidences)].reset_index(drop=True, inplace=False)

# Initialize empty lists
sim_sound = []
sim_evidence = []
sim_mean_ild = []
sim_choice = []

for i in range(n_trials):

    evidence = my_select_evidence(trial_list[i], evidences)  # Select evidence
    sample_index = df_sample[df_sample.Evidence == evidence].index  # Get indexes of sounds with selected evidence
    sound_index = np.random.choice(sample_index)  # Choose a random sound from sample

    # Append values to list
    sim_sound.append(df_sample.Filename.iloc[sound_index])
    sim_evidence.append(df_sample.Evidence.iloc[sound_index])
    sim_mean_ild.append(df_sample.Mean.iloc[sound_index])

    if sim_mean_ild[i] < 0:
        sim_choice.append(0)
    else:
        sim_choice.append(1)

unfair_trials = np.where(np.array(trial_list) != sim_choice)[0]  # Return indices where the choice of the perfect agent
# doesn't match with the known outcome of the trial

# Compute psychometric curves
psych_curve_perfect_agent = compute_psych_curve(sim_evidence, sim_choice)
psych_curve_cheater_agent = compute_psych_curve(sim_evidence, trial_list)

# Plot horizontal and vertical lines
plt.axhline(0.5, color='tab:gray', ls='--')
plt.axvline(0., color='tab:gray', ls='--')

# Plot psychometric curves and errorbars
plt.plot(np.linspace(-1, 1, 30), psych_curve_perfect_agent.fit, color='tab:orange', label='Perfect agent')
plt.errorbar(psych_curve_perfect_agent.xdata, psych_curve_perfect_agent.ydata, yerr=psych_curve_perfect_agent.fit_error,
             color='tab:orange', fmt='o', markerfacecolor='none')

plt.plot(np.linspace(-1, 1, 30), psych_curve_cheater_agent.fit, color='tab:blue', label='Cheater agent')
plt.errorbar(psych_curve_cheater_agent.xdata, psych_curve_cheater_agent.ydata, yerr=psych_curve_cheater_agent.fit_error,
             color='tab:blue', fmt='o', markerfacecolor='none')

plt.title('Psychometric curves \n(' + str(n_trials) + ' trials, ' + str(len(unfair_trials)) + ' unfair)')
# plt.xlabel('Evidence')
plt.xlabel('Interaural Level Difference')
plt.xticks(target_evidences, target_ilds)
plt.ylabel('Probability right')
plt.legend(loc="lower right", frameon=False)
# plt.spines['top'].set_visible(False)
# plt.spines['right'].set_visible(False)


# plt.annotate(str(round(psych_curve.ydata[0], 2)), xy=(psych_curve.xdata[0], psych_curve.ydata[0]),
#               xytext=(psych_curve.xdata[0], psych_curve.ydata[0]), color='tab:red')
# plt.annotate(str(round(psych_curve.ydata[-1], 2)), xy=(psych_curve.xdata[-1], psych_curve.ydata[-1]),
#               xytext=(psych_curve.xdata[-1], psych_curve.ydata[-1]), color='tab:red')
#
# sensitivity, bias, lr_left, lr_right = psych_curve.params
#
# plt.annotate("S=" + str(round(sensitivity, 2)) + "\n" +  # Sensitivity
#               "B=" + str(round(bias, 2)) + "\n" +  # Bias
#               "LR_L=" + str(round(lr_left, 2)) + "\n" +  # Left lapse rate
#               "LR_R=" + str(round(lr_right, 2)), xy=(0, 0), xytext=(-1, 0.5),  # Right lapse rate
#               fontsize='xx-small')









from sympy import symbols, Eq, solve, nsolve, nonlinsolve
from scipy.optimize import fsolve

""""tu primero calculas para amplitud 1 sin poner el 0.00267 porque ya veras que si lo quitas para amplitud 1 no cambia nada practicamente
entonces vas buscando numeros hasta que encuentras uno que te de 73 y cuando ya tienes el 15.535 pones amplitud = 0 y buscas el numero de dentro que te de 33
no me lo he inventado, eh, es el sistema que se usa al ser escalas arbitrarias"""

# Equations to solve
# 73 = x * np.log10((1 + y) / 0.0002)  # 73 = calibration value in dB
# 33 = x * np.log10((0 + y) / 0.0002)  # 33 = ambient noise in dB

73 = A * log( (1 + B) / 0.00002)
33 = A * log( (0 + B) / 0.00002)

73 = x * np.log10((1 + y) / 0.00002)
33 = x * np.log10((0 + y) / 0.00002)


# Approach with scipy's fsolve
def equations(var):
    x, y = var
    eq1 = x * np.log10((1 + y) / 0.00002) - 73
    eq2 = x * np.log10((0 + y) / 0.00002) - 33
    return [eq1, eq2]

x, y =  fsolve(equations, (1, 1), maxfev=10000)
print(equations((x, y)))


# Approach with sympy's nsolve
x, y = symbols('x y')

# Define equations as sympy equation objects
eq1 = Eq(x * np.log10((1 + y) / 0.00002) - 73)
eq2 = Eq(x * np.log10((0 + y) / 0.00002) - 33)

# Solve equations
nsolve((eq1, eq2), (x, y), (1, 1))
nonlinsolve([x * np.log10((1 + y) / 0.00002) - 73, x * np.log10((0 + y) / 0.00002) - 33], (x, y))

nsolve([Eq(x * np.log10((1 + y) / 0.00002) - 73)], [Eq(x * np.log10((0 + y) / 0.00002) - 33)], [x, y], [1, 1])
