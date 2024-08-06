import numpy as np


# 1.

external_current = 200  # nA/mm^2 (I_e/A)
c_m = 10  # nF/cm^2

# Initial values:
V = -65  # mV
m = 0.0529
h = 0.5961
n = 0.3177

# Max conductances
g_L = 0.003  # mS/cm^2
g_K = 0.36  # mS/cm^2
g_Na = 1.2  # mS/cm^2

# Reversal potentials
E_L = -54.387  # mV
E_K = -77  # mV
E_Na = 50  # mV


# Equations:
# n
def alpha_n(V):
    (0.01 * (V + 55)) / (1 - np.exp(-0.1 * (V + 55)))

def beta_n(V):
    0.125 * np.exp(-0.0125 * (V + 65))

# m
def alpha_m(V):
    (0.1 * (V + 40)) / (1 - np.exp(-0.1 * (V + 40)))

def beta_m(V):
    4 * np.exp(-0.0556 * (V + 65))

# h
def alpha_h(V):
    0.07 * np.exp(-0.05 * (V + 65))

def beta_h(V):
    1 / (1 + np.exp(-0.1 * (V + 35)))

def n_inf():
    alpha_n / (alpha_n + beta_n)


# tau_n_dn_dt = (n_inf - n) / tau_n
def tau_n():
    1 / (alpha_n + beta_n)

def dn_over_dt():
    (n_inf - n) / tau_n

def i_m(g_L, g_K, g_Na, V, E_L, E_K, E_Na, m, h, n):
    g_L * (V - E_L) + g_K * n ** 4 * (V - E_K) + g_Na * m ** 3 * h * (V - E_Na)


# c_m_dV_dt = -i_m + external_current
def dV_over_dt():
    (-i_m + external_current) / c_m


time_step = 0.1  # ms

time = np.arange(0, 10, time_step)

for i in time:
    # print(i)

    updated_V = V + dV_over_dt +
    updated_m = m + dV_over_dt +
    updated_h = h + dV_over_dt +
    updated_n = n + dV_over_dt +
