import numpy as np
from unified0d_jlc import *
import matplotlib.pyplot as plt

U = [10, -5, -5]
A = [1, 1, 1]
theta = [np.pi, 0, 0]



theta_vals = np.pi * np.array([16.37922061,
            20.66406614,
            24.94891167,
            28.06516296,
            33.51860273,
            37.80344826,
            42.08829379,
            46.37313932,
            50.65798485,
            54.94283038,
            59.22767591,
            63.51252144,
            67.79736696,
            72.08221249,
            76.36705802,
            80.65190355,
            82.98909202,
            88.44253179,
            92.72737732,
            96.42792573,
            100.1284741,
            104.4133197,
            108.6981652,
            112.9830107,
            117.0730906,
            119.410279])/180
K_calc = 0*theta_vals

for i in range(theta_vals.size):
    theta_val = theta_vals[i]
    theta[1] = theta_val; theta[2] = -theta_val
    C, K, etf = junction_loss_coeff(U, A, theta)
    K_calc[i] = K[0]
K_vals = [0.1776923429,
            0.1917304484,
            0.2102016399,
            0.2199544291,
            0.2552713473,
            0.2858103839,
            0.3146254427,
            0.3478735874,
            0.3845696879,
            0.4247137442,
            0.4665817783,
            0.5109126379,
            0.5579526057,
            0.6069628338,
            0.6576970399,
            0.7091700936,
            0.7303503932,
            0.812116201,
            0.8687611884,
            0.9199879595,
            0.9714610132,
            1.031061391,
            1.093370877,
            1.152971255,
            1.211758901,
            1.246435484]

plt.clf()
plt.scatter(theta_vals, K_calc, label = "calculated")
plt.scatter(theta_vals, K_vals, label = "digitized")
plt.legend(); plt.xlabel("theta"); plt.ylabel("K")
