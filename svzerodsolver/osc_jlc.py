import numpy as np
import pdb
import matplotlib
import matplotlib.pyplot as plt
from junction_loss_coeff import junction_loss_coeff
font = {'family' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

Q = np.asarray([ 6.870e+01, -2.140e+00, -5.461e+01, -1.199e+01,  3.879e-02])
A = np.asarray([1.8005037562382757, 0.22338260436499313, 0.757335625854211, 0.45872861978783136, 0.24982113557391547])
th = np.asarray([3.141592653589793, 3.7076182373395654, 4.580495376727307, 4.167742989473278, 3.327818117607167])
U = np.divide(Q,A)
u4_vec = np.linspace(-2,1,1000)
dP4_vec = 0*u4_vec
for i in range(u4_vec.size):
    U[4] = u4_vec[i]
    C, K, energy_transfer_factor = junction_loss_coeff(U, A, th)
    print(C[-1])
    rho = 1.06
    dP4 = (rho*np.multiply(C[-1], np.square(U[-1]))) + 0.5*rho*np.subtract(
        np.square(U[0]), np.square(U[-1])) # compute pressure loss according to the unified 0d model
    dP4_vec[i] = dP4

plt.clf()
plt.plot(u4_vec, dP4_vec)
plt.xlabel("branch 4 velocity"); plt.ylabel("branch 4 pressure drop")
plt.savefig("vel_vs_dp.png")
