# coding=utf-8

# Copyright (c) Stanford University, The Regents of the University of
#               California, and others.
#
# All Rights Reserved.
#
# See Copyright-SimVascular.txt for additional details.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import scipy
import scipy.sparse.linalg
from scipy.sparse import csr_matrix
import pdb
import copy

try:
    import matplotlib.pyplot as plt
except:
    pass

class GenAlpha:
    """
    Solves system E*ydot + F*y + C = 0 with generalized alpha and Newton-Raphson for non-linear residual
    """
    def __init__(self, rho, y):
        # Constants for generalized alpha
        self.alpha_m = 0.5 * (3.0 - rho) / (1.0 + rho)
        self.alpha_f = 1.0 / (1.0 + rho)
        self.gamma = 0.5 + self.alpha_m - self.alpha_f

        # problem dimension
        self.n = y.shape[0]

        # stores matrices E, F, vector C, and tangent matrices dE, dF, dC
        self.mat = {}

        # jacobian matrix
        self.M = []

        # residual vector
        self.res = []

        # initialize matrices in self.mat
        self.initialize_solution_matrices()

    def initialize_solution_matrices(self):
        """
        Create empty dense matrices and vectors
        """
        mats = ['E', 'F', 'dE', 'dF', 'dC']
        vecs = ['C']

        for m in mats:
            self.mat[m] = np.zeros((self.n, self.n))
        for v in vecs:
            self.mat[v] = np.zeros(self.n)

        # Natalia Addition

    def assemble_structures(self, block_list):
        """
        Assemble block matrices into global matrices
        """
        for bl in block_list:
            for n in self.mat.keys():
                # vectors
                if len(self.mat[n].shape) == 1:
                    for i in range(len(bl.mat[n])):
                        self.mat[n][bl.global_row_id[i]] = bl.mat[n][i]
                # matrices
                else:
                    for i in range(len(bl.mat[n])):
                        for j in range(len(bl.mat[n][i])):
                          try:
                            self.mat[n][bl.global_row_id[i], bl.global_col_id[j]] = bl.mat[n][i][j]
                          except:
                            pdb.set_trace()

    def form_matrix_NR(self, dt):
        """
        Create Jacobian matrix
        """
        self.M = (self.mat['F'] + (self.mat['dE'] + self.mat['dF'] + self.mat['dC'] + self.mat['E'] * self.alpha_m / (
                    self.alpha_f * self.gamma * dt)))

    def form_rhs_NR(self, y, ydot):
        """
        Create residual vector
        """
        self.res = - np.dot(self.mat['E'], ydot) - np.dot(self.mat['F'], y) - self.mat['C']
        # return - csr_matrix(E).dot(ydot) - csr_matrix(F).dot(y) - C

    def form_matrix_NR_numerical(self, res_i, ydotam, args, block_list, epsilon):
        """
        Numerically compute the Jacobian by computing the partial derivatives of the residual using forward finite differences
        """
        # save original values for restoration later
        yaf_original = copy.deepcopy(args['Solution']) # yaf_i

        # compute numerical Jacobian
        J_numerical = np.zeros((self.n, self.n))
        for jj in range(self.n):

            yaf_step_size = np.zeros(self.n)
            yaf_step_size[jj] = np.abs(yaf_original[jj])  * epsilon
            #print("step size: " + str(np.abs(yaf_original[jj])))
            # get solution at the i+1 step
            args['Solution'] = yaf_original  + yaf_step_size # yaf_ip1

            for b in block_list:
                b.update_solution(args)
            self.initialize_solution_matrices()
            self.assemble_structures(block_list)
            self.form_rhs_NR(args['Solution'], ydotam)

            # use forward finite differences (multiply by -1 b/c form_rhs_NR creates the negative residual)
            J_numerical[:, jj] = (self.res - res_i) / yaf_step_size[jj] * -1
        #pdb.set_trace()
        # restore original quantities
        args['Solution'] = yaf_original

        for b in block_list:
            b.update_solution(args)
        self.initialize_solution_matrices()
        self.assemble_structures(block_list)
        self.form_rhs_NR(args['Solution'], ydotam)

        return J_numerical

    def check_jacobian(self, res_i, ydotam, args, block_list):
        """
        Check if the analytical Jacobian (computed from form_matrix_NR) matches the numerical Jacobian
        """
        epsilon_list = np.power(10, np.linspace(-10, 10, 50))
        fig, axs = plt.subplots(2,6, figsize = (20, 20))
        for epsilon in epsilon_list:
            J_numerical = self.form_matrix_NR_numerical(res_i, ydotam, args, block_list, epsilon)
            error = np.abs(self.M - J_numerical)
            #pdb.set_trace()
            for ii in range(2):
                for jj in range(6):
                    if self.mat['dC'][ii, jj] != 0.0:
                        axs[ii, jj].loglog(epsilon, error[ii, jj], 'k*-')

        for ax in axs.flat:
            ax.set(xlabel='epsilon', ylabel='error')
            ax.grid()
            ax.set_aspect('equal', 'box')

        fig.suptitle('absolute error vs epsilon')
        plt.savefig("check_jacobian_plot.png")
        plt.show()

    def visualize_solution(y):
        pass

    def take_newt_step(self, y_new, ydotam_new, args, block_list, dt):
        args["Solution"] = y_new
        for b in block_list:
            b.update_solution(args)
        self.assemble_structures(block_list); self.form_rhs_NR(y_new, ydotam_new); self.form_matrix_NR(dt)
        res_new = np.max(np.abs(copy.deepcopy(self.res)))
        return res_new

    def step(self, y, ydot, t, block_list, args, dt, nit=1e5):
        """
        Perform one time step
        """
        np.set_printoptions(precision=3)

        curr_y = y.copy() + 0.5 * dt * ydot # initial guess for time step
        curr_ydot = ydot.copy() * ((self.gamma - 0.5) / self.gamma) # initial guess for time step

        yaf = y + self.alpha_f * (curr_y - y) # Substep level quantities
        ydotam = ydot + self.alpha_m * (curr_ydot - ydot) # Substep level quantities

        args['Time'] = t + self.alpha_f * dt # initialize solution
        args['Solution'] = yaf # initialize solution

        # initialize blocks
        for b in block_list:
            b.update_constant()
            b.update_time(args)
            b.update_solution(args)

        iit = 0; nit = 30; alpha = 0.1; beta = 0.5;# set Newton method parameters
        self.assemble_structures(block_list); self.form_rhs_NR(yaf, ydotam); self.form_matrix_NR(dt)
        res_old = np.max(np.abs(copy.deepcopy(self.res))); res_trial = copy.deepcopy(res_old)
        dy = scipy.sparse.linalg.spsolve(csr_matrix(self.M), self.res)

        while np.max(np.abs(self.res)) > 5e-4 and iit < nit:
            # Backtracking Algorithm
            # if iit == 0:
            #     y_new= yaf + dy
            # else:
            backtracking = True
            if backtracking:
                ss = 2
                while np.max(np.abs(res_trial)) >= (1-alpha*ss)*np.max(np.abs(res_old)):
                    ss = ss*beta
                    if ss < 1:  print(f"Backtracking.  Current residual {res_old}, trial residual {res_trial}")
                    y_trial = yaf + ss * dy
                    ydotam_trial = ydotam + self.alpha_m * (ss*dy) / (self.alpha_f * self.gamma * dt)
                    res_trial = self.take_newt_step(y_trial, ydotam_trial, args, block_list, dt)
                    # for b in block_list:
                    #     if b.name[0] == "J":
                    #         curr_y, wire_dict, Q, areas, max_inlet = b.unpack_params(args)
                    #         print(f"{b.name[0:2]}: Q = {Q}")#. A: {areas}")
                    # pdb.set_trace()

            res_old = res_trial
            if iit > 50: pdb.set_trace()
            yaf = copy.deepcopy(y_trial); ydotam = copy.deepcopy(ydotam_trial); res_old = copy.deepcopy(res_trial)
            dy = scipy.sparse.linalg.spsolve(csr_matrix(self.M), self.res)
            if np.any(np.isnan(self.res)):
                raise RuntimeError('Solution nan')
            iit += 1
            print(iit, " Newton iterations. Current residual: " , np.max(np.abs(self.res)))

        if iit >= nit:
            print("Max NR iterations (" ,iit,") reached at time: ", t, " , max error: ", max(abs(self.res)))
            curr_y = y + (yaf - y) / self.alpha_f
        else:
            print("timestep ", t, " completed.  ", iit, " Newton iterations.")
            # update time step
            curr_y = y + (yaf - y) / self.alpha_f
            curr_ydot = ydot + (ydotam - ydot) / self.alpha_m

        args['Time'] = t + dt

        return curr_y, curr_ydot
