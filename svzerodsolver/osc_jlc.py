import numpy as np
import pdb
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from junction_loss_coeff import junction_loss_coeff
from junction_loss_coeff_tf import junction_loss_coeff_tf
font = {'family' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

# Q = np.asarray([ 6.870e+01, -2.140e+00, -5.461e+01, -1.199e+01,  3.879e-02])
# A = np.asarray([1.8005037562382757, 0.22338260436499313, 0.757335625854211, 0.45872861978783136, 0.24982113557391547])
# th = np.asarray([3.141592653589793, 3.7076182373395654, 4.580495376727307, 4.167742989473278, 3.327818117607167])
# U = np.divide(Q,A)
# u4_vec = np.linspace(-2,1,1000)
# dP4_vec = 0*u4_vec
# for i in range(u4_vec.size):
#     U[4] = u4_vec[i]
#     C, K, energy_transfer_factor = junction_loss_coeff(U, A, th)
#     print(C[-1])
#     rho = 1.06
#     dP4 = (rho*np.multiply(C[-1], np.square(U[-1]))) + 0.5*rho*np.subtract(
#         np.square(U[0]), np.square(U[-1])) # compute pressure loss according to the unified 0d model
#     dP4_vec[i] = dP4

# Q = np.asarray([ 5.008e+00, -5.012e+00,  3.584e-03])
# A = np.asarray([0.4911395121641283, 0.37717405151945504, 0.18840448761130651])
# th = np.asarray([3.141592653589793, 3.2601554192117788, 3.7198101791990705])
# u2_vec = np.linspace(-1,1,1000)
# dP2_vec = 0*u2_vec; dP1_vec = 0*u2_vec
# for i in range(u2_vec.size):
#     U[2] = u2_vec[i]
#     C, K, energy_transfer_factor = junction_loss_coeff(U, A, th)
#     print(C[-1])
#     rho = 1.06
#     dP1 = (rho*np.multiply(C[1], np.square(U[1])))
#     dP2 = (rho*np.multiply(C[2], np.square(U[2])))
Q = np.asarray([ 6.644e+00, -6.644e+00, -4.500e-06])
A = np.asarray([0.46216038052378955, 0.3518373931132117, 0.1839880185439387])
th = np.asarray([3.141592653589793, 3.209623197055208, 3.750934586167999])

U = np.divide(Q,A)


u2_vec = np.linspace(-2,2,1000)
dP2_vec = 0*u2_vec; dP1_vec = 0*u2_vec

for i in range(u2_vec.size):
    U[-1] = u2_vec[i]
    C, K, energy_transfer_factor = junction_loss_coeff(U, A, th)
    #print(C[-1])
    rho = 1.06
    # dP_unified0d_outlets = (rho*np.multiply(C, np.square(U))) + 0.5*rho*np.subtract(
    #     np.square(U[0]), np.square(U)) # compute pressure loss according to the unified 0d model
    dP_unified0d_outlets = (rho*np.multiply(C, np.square(U)))
    dP1 = dP_unified0d_outlets[1]; dP2 = dP_unified0d_outlets[2];
    dP1_vec[i] = dP1; dP2_vec[i] = dP2
    pdb.set_trace()




plt.clf()
plt.plot(u2_vec, dP1_vec, label=("outlet 1"))
plt.plot(u2_vec, dP2_vec, label=("outlet 2"))
plt.xlabel("branch 2 velocity"); plt.ylabel("dP"); plt.legend()
plt.savefig("vel_vs_dp_py.png")

pdb.set_trace()
