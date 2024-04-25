import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def lorenz(w, t, o, p, b):
    x, y, z = w
    dxdt = o * (y - x)
    dydt = x * (p - z) - y
    dzdt = x*y -b*z
    return [dxdt, dydt, dzdt]

def lorenz_reconstructed(w, t, o, p ,b):
    x, y, z = w
    dSdt = -9.96*x + 9.96*y
    dIdt = 25.438*x -0.946*x*z
    dRdt = -2.652*z + 0.995*x*y
    return [dSdt, dIdt, dRdt]

def simulate(equations, t):
    o = 10
    p = 28
    b = 8/3
    w0 = [1, 1, 1]
    sol = odeint(equations, w0, t, args=(o, p, b))
    return sol

def test():
    t = np.linspace(0, 30, 2000)
    sol1 = simulate(lorenz, t)
    sol2 = simulate(lorenz_reconstructed, t)

    ax = plt.axes(projection='3d')
    ax.plot3D(sol1[:, 0],sol1[:, 1],sol1[:, 2], 'orange')
    ax.plot3D(sol2[:, 0],sol2[:, 1],sol2[:, 2], 'green')
    plt.legend(loc='best')
    plt.show()

#test()