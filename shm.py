import numpy as np
import pandas as pd


def newmann_beta(m, c, k, dt, f, number_of_time_points):

    # Initialise
    x = np.zeros(number_of_time_points)
    v = np.zeros(number_of_time_points)
    a = np.zeros(number_of_time_points)

    # Initial Conditions
    x[0] = 0
    v[0] = 0
    a[0] = f[0] / m

    for i in range(1, number_of_time_points):
        df = f[i] - f[i-1]
        kk = ((6.0 * m) / (dt**2)) + ((3.0 * c) / dt) + k
        ff = df + ((3.0 * m) + (dt / 2.0 * c)) * a[i-1] + ((6.0 / dt * m) + (3.0 * c)) * v[i-1]
        dx = ff / kk

        x[i] = dx + x[i-1]
        v[i] = (-2.0 * v[i-1]) - (dt / 2.0 * a[i-1]) + (3.0 / dt * dx)
        a[i] = (-1.0 / m) * ((c * v[i]) + (k * x[i]) - f[i])

    cols = ['exc', 'dis', 'vel', 'acc']
    data = dict(zip(cols, [f, x, v, a]))
    df = pd.DataFrame(data)

    return df
