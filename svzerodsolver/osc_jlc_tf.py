import numpy as np
import pdb
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
#from junction_loss_coeff import junction_loss_coeff
from junction_loss_coeff_tf_3_18 import junction_loss_coeff_tf
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
# q2_vec = np.linspace(-1,1,1000)
# dP2_vec = 0*q2_vec; dP1_vec = 0*q2_vec
# for i in range(q2_vec.size):
#     U[2] = q2_vec[i]
#     C, K, energy_transfer_factor = junction_loss_coeff(U, A, th)
#     print(C[-1])
#     rho = 1.06
#     dP1 = (rho*np.multiply(C[1], np.square(U[1])))
#     dP2 = (rho*np.multiply(C[2], np.square(U[2])))
Q = np.asarray([ 6.644e+00, -6.644e+00, -4.500e-06])
areas = np.asarray([0.46216038052378955, 0.3518373931132117, 0.1839880185439387])
angles = np.asarray([3.141592653589793, 3.209623197055208, 3.750934586167999])

Q_tensor = tf.Variable(tf.convert_to_tensor(Q)) # convert Q to a tensor
areas_tensor = tf.convert_to_tensor(areas, dtype="double") # convert areas to a tensor
angles_tensor = tf.convert_to_tensor(angles, dtype="double") # convert angles to a tensor
#U = np.divide(Q,A)

@tf.function
def unified0d_tf(Q_tensor, areas_tensor, angles_tensor):
    rho = 1.06 # density of blood
    with tf.GradientTape(watch_accessed_variables=False, persistent=False) as tape:
      tape.watch(Q_tensor) # track operations applied to Q_tensor
      U_tensor = tf.divide(Q_tensor, areas_tensor)
      C_outlets, outlets = junction_loss_coeff_tf(U_tensor, areas_tensor, angles_tensor) # run Unified0D junction loss coefficient function
      dP_unified0d_outlets = (rho*tf.multiply(
          C_outlets, tf.square(tf.boolean_mask(U_tensor, outlets))))
    ddP_dU_outlets = tape.jacobian(dP_unified0d_outlets, Q_tensor) # get derivatives of pressure loss coeff wrt. U_tensor
    return dP_unified0d_outlets, ddP_dU_outlets # get derivatives of pressure loss coeff wrt. U_tensor

unified_tf_concrete = unified0d_tf.get_concrete_function(Q_tensor, areas_tensor, angles_tensor)

q2_vec = np.linspace(-2,2,1000)*areas[-1]
dP2_vec = 0*q2_vec; dP1_vec = 0*q2_vec
ddP10_vec = 0*q2_vec; ddP11_vec = 0*q2_vec; ddP12_vec = 0*q2_vec;
ddP20_vec = 0*q2_vec; ddP21_vec = 0*q2_vec; ddP22_vec = 0*q2_vec
for i in range(q2_vec.size):
    Q[2] = q2_vec[i]; Q_tensor = tf.Variable(tf.convert_to_tensor(Q)); print(Q[2])
    dP_unified0d_outlets, ddP_dU_outlets = unified_tf_concrete(Q_tensor, areas_tensor, angles_tensor)
    # U_tensor = tf.divide(Q_tensor, areas_tensor); pdb.set_trace()
    # C_outlets, outlets = junction_loss_coeff_tf(U_tensor, areas_tensor, angles_tensor) # run Unified0D junction loss coefficient function
    # dP_unified0d_outlets = (rho*tf.multiply(
    #       C_outlets, tf.square(tf.boolean_mask(U_tensor, outlets))))

    dP_unified0d_outlets = dP_unified0d_outlets.numpy()
    #pdb.set_trace()
    ddP_dU_outlets = ddP_dU_outlets.numpy()
    # dP_unified0d_outlets, ddP_dU_outlets = unified0d_tf(Q_tensor, areas_tensor, angles_tensor)
    # dP_unified0d_outlets = dP_unified0d_outlets.numpy()
    #pdb.set_trace()
    #ddP_dU_outlets = ddP_dU_outlets.numpy()
    if dP_unified0d_outlets.size == 2:
        dP1 = dP_unified0d_outlets[0];
        ddP10 = ddP_dU_outlets[0][0]; ddP11 = ddP_dU_outlets[0][1]; ddP12 = ddP_dU_outlets[0][2]
        dP2 = dP_unified0d_outlets[1];
        ddP20 = ddP_dU_outlets[1][0]; ddP21 = ddP_dU_outlets[1][1]; ddP22 = ddP_dU_outlets[1][2]
    else:
        dP1 = dP_unified0d_outlets[0];
        ddP10 = ddP_dU_outlets[0][0]; ddP11 = ddP_dU_outlets[0][1]; ddP12 = ddP_dU_outlets[0][2]
        dP2 = 0;
        ddP20 = 0; ddP21 = 0; ddP22 = 0
    dP1_vec[i] = dP1; dP2_vec[i] = dP2
    #pdb.set_trace()
    ddP10_vec[i] = ddP10; ddP11_vec[i] = ddP11; ddP12_vec[i] = ddP12;
    ddP20_vec[i] = ddP20; ddP21_vec[i] = ddP21; ddP22_vec[i] = ddP22;

plt.clf()
plt.plot(q2_vec/areas[-1], dP1_vec, label=("outlet 1"))
plt.plot(q2_vec/areas[-1], dP2_vec, label=("outlet 2"))
plt.xlabel("branch 2 velocity"); plt.ylabel("dP"); plt.legend()
plt.savefig("vel_vs_dp_tf.png")

plt.clf()
plt.plot(q2_vec, ddP10_vec, label=("dP1/dU0"), linewidth=2, color = "sandybrown")
plt.plot(q2_vec, ddP11_vec, label=("dP1/dU1"), linewidth=2, color = "tomato")
plt.plot(q2_vec, ddP12_vec, label=("dP1/dU2"), linewidth=2, color = "lightpink")
plt.plot(q2_vec, ddP20_vec, label=("dP2/dU0"), linewidth=2, color = "palegreen")
plt.plot(q2_vec, ddP21_vec, label=("dP2/dU1"), linewidth=2, color = "mediumturquoise")
plt.plot(q2_vec, ddP22_vec, label=("dP2/dU2"), linewidth=2, color = "royalblue")

plt.xlabel("branch 2 velocity"); plt.ylabel("ddP/dQ"); plt.legend()
plt.savefig("vel_vs_ddp_tf.png")
pdb.set_trace()
