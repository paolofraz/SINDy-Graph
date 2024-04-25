import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

def SIR(y, t, b, g):
    S, I, R = y
    N = S+I+R
    dSdt = -b * I * S / N
    dIdt = b * I * S / N - g * I
    dRdt = g * I
    return [dSdt, dIdt, dRdt]

def simulate(y0, equations, t):
    b = 0.04
    g = 0.02
    sol = odeint(equations, y0, t, args=(b, g))
    return sol

def test():
    t = np.linspace(0, 10, 100)
    sol1 = simulate(SIR, t)
    sol2 = simulate(SIR_reconstructed, t)
    plt.plot(t, sol1[:, 0], label='S')
    plt.plot(t, sol1[:, 1], label='I')
    plt.plot(t, sol1[:, 2], label='R')
    plt.plot(t, sol2[:, 0], label='S_rec')
    plt.plot(t, sol2[:, 1], label='I_rec')
    plt.plot(t, sol2[:, 2], label='R_rec')
    plt.legend(loc='best')
    plt.show()
