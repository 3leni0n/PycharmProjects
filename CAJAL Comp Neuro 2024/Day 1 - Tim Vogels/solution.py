import numpy as np
import matplotlib.pyplot as plt
# Variables
duration = 1000
dt = 0.1
time = np.arange(0, duration + dt, dt)
cm = 10
gogo = np.arange(0, 501) # number of different current injections
# Vectors
m = np.zeros(len(time))
h = np.zeros(len(time))
n = np.zeros(len(time))
V = np.zeros(len(time))
Vinf = np.zeros(len(time))
alphaN = np.zeros(len(time))
alphaM = np.zeros(len(time))
alphaH = np.zeros(len(time))
betaM = np.zeros(len(time))
betaN = np.zeros(len(time))
betaH = np.zeros(len(time))
Ie = np.zeros((len(gogo), len(time)))
# Initials
m[0] = 0.0529
h[0] = 0.5961
n[0] = 0.3177
gleak = 0.003 * 1000 # factor of 1000 because units
gK = 0.36 * 1000
gNA = 1.2 * 1000
Eleak = -54.387
EK = -77
ENA = 50
V[0] = -65
# Vinf[0] = -65
# Ie[:int(1000/dt)] = 1
# Ie[int(100/dt):int(105/dt)] = -50
tslc = 0
for k in range(len(gogo)):
    Ie[k, :] = gogo[k]
rate = np.zeros(len(gogo))
# Code
for k in range(len(gogo)):
    for i in range(len(time) - 1):
        Gtotal = gleak + (gK * (n[i] ** 4)) + (gNA * (m[i] ** 3) * h[i])
        taueff = cm / Gtotal
        Vinf[i + 1] = (gleak * Eleak + (gK * (n[i] ** 4) * EK) + (gNA * (m[i] ** 3) * h[i] * ENA) + Ie[k, i]) / Gtotal
        V[i + 1] = Vinf[i + 1] + (V[i] - Vinf[i + 1]) * np.exp(-dt / taueff)
        if V[i + 1] > 0 and tslc <= 0:
            rate[k] += 1
            tslc = 12
        tslc -= 1
        alphaN[i + 1] = (0.01 * (V[i + 1] + 55)) / (1 - np.exp(-0.1 * (V[i + 1] + 55)))
        alphaM[i + 1] = (0.1 * (V[i + 1] + 40)) / (1 - np.exp(-0.1 * (V[i + 1] + 40)))
        alphaH[i + 1] = 0.07 * np.exp(-0.05 * (V[i + 1] + 65))
        betaN[i + 1] = 0.125 * np.exp(-0.0125 * (V[i + 1] + 65))
        betaM[i + 1] = 4 * np.exp(-0.0556 * (V[i + 1] + 65))
        betaH[i + 1] = 1 / (1 + np.exp(-0.1 * (V[i + 1] + 35)))
        tauN = 1 / (alphaN[i + 1] + betaN[i + 1])
        tauM = 1 / (alphaM[i + 1] + betaM[i + 1])
        tauH = 1 / (alphaH[i + 1] + betaH[i + 1])
        ninf = alphaN[i + 1] / (alphaN[i + 1] + betaN[i + 1])
        n[i + 1] = ninf + (n[i] - ninf) * np.exp(-dt / tauN)
        minf = alphaM[i + 1] / (alphaM[i + 1] + betaM[i + 1])
        m[i + 1] = minf + (m[i] - minf) * np.exp(-dt / tauM)
        hinf = alphaH[i + 1] / (alphaH[i + 1] + betaH[i + 1])
        h[i + 1] = hinf + (h[i] - hinf) * np.exp(-dt / tauH)
# Plots
plt.figure(1)
plt.subplot(2, 1, 1)
plt.ylabel('V[mV]')
plt.plot(time, V)
plt.plot(time, Vinf, 'r')
plt.figure(3)
plt.plot(gogo, rate, '.')
plt.xlabel('gogo (input injected)')
plt.ylabel('rate')
plt.show()