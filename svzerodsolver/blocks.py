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
import copy
import pdb
import numpy as np
from collections import defaultdict
#from junction_loss_coeff import junction_loss_coeff
from .junction_loss_coeff_tf_3_18 import junction_loss_coeff_tf
import tensorflow as tf
import time

class LPNVariable:
    def __init__(self, value, units, name="NoName", vtype='ArbitraryVariable'):
        self.type = vtype
        self.value = value
        # Two generic units accepted : SI, cgs. Conversion for special values applied
        self.units = units
        self.name = name

class PressureVariable(LPNVariable):

    def __init__(self, value, units='cgs', name='NoNamePressure'):
        LPNVariable.__init__(self, value=value, units=units, name=name, vtype='Pressure')

    def convert_to_cgs(self):
        if self.units == 'cgs':
            print("Variable: " + self.name + " already at cgs")
        elif self.units == 'SI':
            self.value *= 1.0E5
            self.units = 'cgs'
        elif self.units == 'mmHg':
            self.value *= 0.001333224
            self.units = 'cgs'
        else:
            raise Exception("Units " + self.units + " not recognized")

    def convert_to_mmHg(self):
        if self.units == 'cgs':
            self.value *= 750.06
            self.units = 'mmHg'
        elif self.units == 'SI':
            self.value = self.value * 7.50 * 1E-3
            self.units = 'mmHg'
        elif self.units == 'mmHg':
            print("Variable: " + self.name + " already at mmHg")
        else:
            raise Exception("Units " + self.units + " not recognized")

class FlowVariable(LPNVariable):

    def __init__(self, value, units='cgs', name='NoNameFlow'):
        LPNVariable.__init__(self, value=value, units=units, name=name, vtype='Flow')

    def convert_to_cgs(self):
        if self.units == 'cgs':
            print("Variable: " + self.name + " already at cgs")
        elif self.units == 'SI':
            self.value = self.value * 1.0E-6
            self.units = 'cgs'
        elif self.units == 'Lpm':  # litres per minute
            self.value = self.value * 16.6667
            self.units = 'cgs'
        else:
            raise Exception("Units " + self.units + " not recognized")

    def convert_to_Lpm(self):
        if self.units == 'cgs':
            self.value = self.value / 16.6667
            self.units = 'Lpm'
        elif self.units == 'SI':
            self.value = self.value / (16.6667 * 1.0E-6)
            self.units = 'Lpm'
        elif self.units == 'Lpm':
            print("Variable: " + self.name + " already at Lpm")
        else:
            raise Exception("Units " + self.units + " not recognized")

class wire:
    """
    Wires connect circuit elements and junctions
    They can only posses a single pressure and flow value (system variables)
    They can also only possess one element(or junction) at each end
    """
    def __init__(self, connecting_elements, Pval=0, Qval=0, name="NoNameWire", P_units='cgs', Q_units='cgs'):
        self.name = name
        self.type = 'Wire'
        self.P = PressureVariable(value=Pval, units=P_units, name=name + "_P")
        self.Q = FlowVariable(value=Qval, units=Q_units, name=name + "_Q")
        if len(connecting_elements) > 2:
            raise Exception('Wire cannot connect to more than two elements at a time. Use a junction LPN block')
        if type(connecting_elements) != tuple:
            raise Exception('Connecting elements to wire should be passed as a 2-tuple')
        self.connecting_elements = connecting_elements
        self.LPN_solution_ids = [None] * 2

class LPNBlock:
    def __init__(self, connecting_block_list=None, name="NoName", flow_directions=[]):
        if connecting_block_list == None:
            connecting_block_list = []
        self.connecting_block_list = connecting_block_list
        self.num_connections = len(connecting_block_list)
        self.name = name
        self.neq = 2
        self.n_connect = 2
        self.n_connect = None
        self.type = "ArbitraryBlock"
        self.num_block_vars = 0
        self.connecting_wires_list = []

        # -1 : Inflow to block, +1 outflow from block
        self.flow_directions = flow_directions

        # solution IDs for the LPN block's internal solution variables
        self.LPN_solution_ids = []

        # block matrices
        self.mat = defaultdict(list)

        # row and column indices of block in global matrix
        self.global_col_id = []
        self.global_row_id = []

    def check_block_consistency(self):
        if len(connecting_block_list) != self.n_connect:
            msg = self.name + " block can be connected only to " + str(self.n_connect) + " elements"
            raise Exception(msg)

    def add_connecting_block(self, block, direction):
        # Direction = +1 if flow sent to block
        #            = -1 if flow recvd from block
        self.connecting_block_list.append(block)
        self.num_connections = len(self.connecting_block_list)
        self.flow_directions.append(direction)

    def add_connecting_wire(self, new_wire):
        self.connecting_wires_list.append(new_wire)

    def update_constant(self):
        """
        Update solution- and time-independent blocks
        """
        pass

    def update_time(self, args):
        """
        Update time-dependent blocks
        """
        pass

    def update_solution(self, args):
        """
        Update solution-dependent blocks
        """
        pass

    def eqids(self, wire_dict, local_eq):
        # EqID returns variable's location in solution vector

        nwirevars = self.num_connections * 2  # num_connections is multipled by 2 because each wire has 2 soltns (P and Q)
        if local_eq < nwirevars:
            vtype = local_eq % 2  # 0 --> P, 1 --> Q
            wnum = int(local_eq / 2)

            # example: assume num_connections is 2. this can be a normal resistor block, which has 2 connections. then this R block has 2 connecting wires. thus, this R block has 4 related solution variables/unknowns (P_in, Q_in, P_out, Q_out). note that local_eq = local ID.
            #     then for these are the vtypes we get for each local_eq:
            #         local_eq    :     vtype     :     wnum
            #         0            :     0        :    0        <---    vtype = pressure, wnum = inlet wire
            #         1            :    1        :    0        <---    vtype = flow, wnum = inlet wire
            #         2            :    0        :    1        <---    vtype = pressure, wnum = outlet wire
            #         3            :    1        :    1        <---    vtype = flow, wnum = outlet wire
            #    note that vtype represents whether the solution variable in local_eq (local ID) is a P or Q solution
            #        and wnum represents whether the solution variable in local_eq comes from the inlet wire or the outlet wire, for this LPNBlock with 2 connections (one inlet, one outlet)

            return wire_dict[self.connecting_wires_list[wnum]].LPN_solution_ids[vtype]
        else:  # this section will return the index at which the LPNBlock's  INTERNAL SOLUTION VARIABLES are stored in the global vector of solution unknowns/variables (i.e. I think RCR and OpenLoopCoronaryBlock have internal solution variables; these internal solution variables arent the P_in, Q_in, P_out, Q_out that correspond to the solutions on the attached wires, they are the solutions that are internal to the LPNBlock itself)
            vnum = local_eq - nwirevars
            return self.LPN_solution_ids[vnum]

    def form_derivative_num(self, args, epsilon):
      #print("form deriv running")
      #pdb.set_trace()
      self.update_solution(args)

      #y_global = args['Solution']  # the current solution for all unknowns in our 0D model
      wire_dict = args['Wire dictionary'] # connectivity dictionary
      y_block_indices = []
      for wire in self.connecting_wires_list:
        y_block_indices = y_block_indices + wire_dict[wire].LPN_solution_ids

      C_original = self.mat["C"]
      dC_analytical = self.mat["dC"]
      if C_original and dC_analytical:
        #pdb.set_trace()

        preturbed_args = copy.copy(args)
        dC_numerical = []
        #pdb.set_trace()
        for pret_index in y_block_indices:
          preturbation = np.zeros(len(args["Solution"]))
          #pdb.set_trace()
          preturbation[pret_index] = max(np.abs(args["Solution"][pret_index])  * epsilon, epsilon)
          preturbed_args["Solution"] = args["Solution"]  + preturbation
          #pdb.set_trace()
          self.update_solution(preturbed_args)
          C_preturbed = self.mat["C"]
          dC_numerical.append([(C_preturbed[i]-C_original[i])/preturbation[pret_index] for i in range(len(C_original))])
        dC_numerical = np.transpose(np.asarray(dC_numerical))
        dC_analytical = np.asarray(dC_analytical)
        #print("dC_numerical", dC_numerical)
        #print("dC_analytical", dC_analytical)
        return dC_numerical, dC_analytical
        #pdb.set_trace()


    def check_jacobian(self, args):
        self.update_solution(args)
        C_original = self.mat["C"]
        dC_analytical = np.asarray(self.mat["dC"])
        self.form_derivative_num(args, epsilon)

class Junction(LPNBlock):
    """
    Junction points between LPN blocks with specified directions of flow
    """
    def __init__(self, connecting_block_list=None, name="NoNameJunction", flow_directions=None):
        LPNBlock.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.type = "Junction"
        self.neq = self.num_connections  # number of equations = num of blocks that connect to this junction, where the equations are 1) mass conservation 2) inlet pressures = outlet pressures

    def add_connecting_block(self, block, direction):
        self.connecting_block_list.append(block)
        self.num_connections = len(self.connecting_block_list)
        self.neq = self.num_connections
        self.flow_directions.append(direction)

    def update_constant(self):
        # Number of variables per tuple = 2*num_connections
        # Number of equations = num_connections-1 Pressure equations, 1 flow equation
        # Format : P1,Q1,P2,Q2,P3,Q3, .., Pn,Qm
        self.mat['F'] = [(1.,) + (0,) * (2 * i + 1) + (-1,) + (0,) * (2 * self.num_connections - 2 * i - 3) for i in
                         range(self.num_connections - 1)]

        tmp = (0,)
        for d in self.flow_directions[:-1]:
            tmp += (d,)
            tmp += (0,)

        tmp += (self.flow_directions[-1],)
        self.mat['F'].append(tmp)

class STATICPJunction(Junction):

    def __init__(self, junction_parameters, connecting_block_list=None, name="NoNameJunction", flow_directions=None):
        Junction.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.flow_directions = flow_directions
        self.junction_parameters = junction_parameters
        self.rep_Q_inds = {}
        self.rho = 1.06
        for Q_ind in range(len(self.flow_directions)):
            self.rep_Q_inds.update({Q_ind:[]})

    def unpack_params(self, args):
        curr_y = args['Solution']  # the current solution for all unknowns in our 0D model
        wire_dict = args['Wire dictionary'] # connectivity dictionary
        areas = self.junction_parameters["areas"] # load areas
        Q = np.abs(np.asarray([curr_y[wire_dict[self.connecting_wires_list[i]].LPN_solution_ids[1]] for i in range(len(self.flow_directions))])) # calculate velocity in each
        V = np.divide(Q,areas)
        return curr_y, wire_dict, Q, areas


    def set_F(self):
        self.mat['F'] = [(1.,) + (0,) * (2 * i + 1) + (-1,) + (0,) * (2 * self.num_connections - 2 * i - 3) for i in
                         range(self.num_connections - 1)] # copy-pasted from normal junction
        tmp = (0,)
        for d in self.flow_directions[:-1]:
            tmp += (d,)
            tmp += (0,)
        tmp += (self.flow_directions[-1],)
        self.mat['F'].append(tmp)


    @tf.function
    def static_p_tf(self, Q_tf, areas_tf):
        rho = tf.constant(1.06, dtype=tf.float32) # density of blood
        with tf.GradientTape(watch_accessed_variables=False, persistent=False) as tape:
          tape.watch(Q_tf) # track operations applied to Q_tensor
          V = tf.math.divide(Q_tf,areas_tf)
          C = tf.multiply(rho, tf.multiply(tf.constant(0.5, dtype=tf.float32), tf.subtract(
          tf.math.square(V), tf.math.square(tf.slice(V, begin = [0], size = [1]))
          )))

        dC_dQ = tape.jacobian(C, Q_tf) # get derivatives of pressure loss coeff wrt. U_tensor
        return C, dC_dQ

    def set_C(self, args, Q, areas):

        Q_tf = tf.Variable(tf.convert_to_tensor(Q, dtype=tf.float32)) # convert Q to a tensor
        areas_tf = tf.convert_to_tensor(areas, dtype=tf.float32) # convert areas to a tensor

        branch_config = Q.size
        if branch_config in args["tf_graph_dict"]:
          unified_tf_concrete = args["tf_graph_dict"][branch_config]
        else:
          unified_tf_concrete = self.static_p_tf.get_concrete_function(Q_tf, areas_tf)
          args["tf_graph_dict"].update({branch_config: unified_tf_concrete})
          print("Adding tf.concrete_function for junction with "+ str(branch_config) + " branches.")
        C_tf, dC_dQ_tf = unified_tf_concrete(Q_tf, areas_tf)

        self.mat["C"] = list(C_tf.numpy()[1:]) + [0]

        dC_dQ_np = dC_dQ_tf.numpy()
        #pdb.set_trace()
        dC_dsol = []
        for i in range(len(Q)):
            deriv_list = []
            for j in range(len(Q)): # loop over velocity derivatives
                deriv_list.append(0);deriv_list.append(dC_dQ_np[i,j])
            dC_dsol.append(tuple(deriv_list))
        dC_dsol.append(tuple([0*i for i in deriv_list]))
        self.mat["dC"] = dC_dsol[1:]
        return


    def update_solution(self, args):
        curr_y, wire_dict, Q, areas = self.unpack_params(args)

        if np.sum(np.asarray(Q)!=0) == 0: # if all velocities are 0, treat as normal junction (copy-pasted from normal junction code)
            self.set_F() # set F matrix and c vector for zero flow case
            print("all zeros")
        else: # otherwise apply Unified0D model
            self.set_F() # set F matrix
            self.set_C(args, Q, areas)
            #pdb.set_trace()


class UNIFIED0DJunction(Junction):

    def __init__(self, junction_parameters, connecting_block_list=None, name="NoNameJunction", flow_directions=None):
        Junction.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.flow_directions = flow_directions
        self.junction_parameters = junction_parameters
        self.rep_Q_inds = {}
        self.rho = 1.06
        for Q_ind in range(len(self.flow_directions)):
            self.rep_Q_inds.update({Q_ind:[]})
    def unpack_params(self, args):
        curr_y = args['Solution']  # the current solution for all unknowns in our 0D model
        wire_dict = args['Wire dictionary'] # connectivity dictionary
        areas = self.junction_parameters["areas"] # load areas
        Q = np.asarray([-1*self.flow_directions[i] *
            curr_y[wire_dict[self.connecting_wires_list[i]].LPN_solution_ids[1]] for i in range(len(self.flow_directions))]) # calculate velocity in each branch (inlets positive, outlets negative)
        small_Q = np.reshape(np.asarray(np.where(np.abs(Q) <= 0.1)),(-1,))
        # if not np.all(Q==0):
        #     Q[small_Q] = 0
        tangents = copy.deepcopy(self.junction_parameters["tangents"]) # load angles
        return curr_y, wire_dict, Q, areas, tangents, small_Q

    def check_rep_Q(self, Q):
        Q_frozen = copy.deepcopy(Q)
        small_Qs = np.asarray([])
        #small_Qs = np.extract(np.abs(Q)<=(np.max(np.abs(Q))/100), Q)
        if small_Qs.size > 0:
            print("small Qs: ", small_Qs)
        Q_freeze_ind = []
        for small_Q in small_Qs:
            Q_ind = list(Q).index(small_Q)
            self.rep_Q_inds[Q_ind].append(small_Q)
            num_reps = np.count_nonzero(self.rep_Q_inds[Q_ind] == small_Q)
            #num_reps = np.count_nonzero(np.abs(self.rep_Q_inds[Q_ind] - small_Q)< 0.001)
            #num_reps = np.count_nonzero(np.abs((self.rep_Q_inds[Q_ind] - small_Q)/small_Q)< 0.001)
            if num_reps >= 2:
                Q_freeze_ind.append(Q_ind)
                Q_frozen[Q_ind] = 0
                print("freezing Q for branch ", Q_ind)
        return Q_frozen, Q_freeze_ind


    def F_add_continuity_row(self):
        tmp = (0,)
        for d in self.flow_directions[:-1]:
            tmp += (d,)
            tmp += (0,)
        tmp += (self.flow_directions[-1],)
        self.mat['F'].append(tmp)

    def classify_branches(self, Q):
        inlet_indices = list(np.asarray(np.nonzero(Q>=0)).astype(int)[0]) # identify inlets
        Q_in = Q[inlet_indices]
        max_inlet_tol = 0.01
        #pdb.set_trace()
        max_inlet_cands = np.asarray(inlet_indices)[Q_in > (np.max(Q_in)-max_inlet_tol)]
        max_inlet = np.min(max_inlet_cands)
        num_inlets = len(inlet_indices) # number of inlets
        outlet_indices = list(np.nonzero(Q<0)[0]) # identify outlets
        num_outlets = len(outlet_indices) # number of outlets
        num_branches = Q.size # number of branches
        # inlet/outlet sanity checks
        if num_inlets ==0:
          pdb.set_trace()
        assert num_inlets != 0, "No junction inlet."
        assert num_outlets !=0, "No junction outlet. Q: " + str(Q)
        assert num_inlets + num_outlets == num_branches, "Sum of inlets and outlets does not equal the number of branches."
        return inlet_indices, max_inlet, num_inlets, outlet_indices, num_outlets, num_branches

    def set_no_flow_mats(self):
        self.mat['F'] = [(1.,) + (0,) * (2 * i + 1) + (-1,) + (0,) * (2 * self.num_connections - 2 * i - 3) for i in
                         range(self.num_connections - 1)] # copy-pasted from normal junction
        self.F_add_continuity_row() # add mass conservation row

    def set_F(self, Q):
        inlet_indices, max_inlet, num_inlets, outlet_indices, num_outlets, num_branches = self.classify_branches(Q)
        self.mat['F'] = []
        for i in range(0,num_branches): # loop over branches- each branch (exept the "presumed inlet" is a row of F)
            if i == max_inlet:
              continue # if the branch is the dominant inlet branch do not add a new column
            F_row = [0]*(2*num_branches) # row of 0s with 1 in column corresponding to "presumed inlet" branch pressure
            F_row[2*max_inlet] = 1 # place 1 in column corresponding to dominant inlet pressure
            F_row[2*i] = -1 # place -1 in column corresponding to each branch pressure
            self.mat['F'].append(tuple(F_row)) # append row to F matrix
        self.F_add_continuity_row() # add mass conservation row

    def get_angle_difference(self, tangent1, tangent2):
        angle_difference = np.arccos(np.sum(np.multiply(
                            np.asarray(tangent1),
                            np.asarray(tangent2)), axis=0))
        return angle_difference

    def configure_angles(self, tangents, Q, max_inlet, num_branches):
        angles = []
        for i in range(num_branches): # loop over all angles
            if i == max_inlet:
                 angles.append(np.pi)
            else:
                if Q[i] < 0:
                    angles.append(self.get_angle_difference(tangents[max_inlet], tangents[i]))
                else:
                    angles.append(np.pi + self.get_angle_difference(tangents[max_inlet], tangents[i]))
        assert len(angles) == num_branches, 'One angle should be provided for each branch'
        #print(angles)
        return angles

    @tf.function
    def unified0d_tf(self, Q_tensor, areas_tensor, angles_tensor, num_outlets, outlet_indices, max_inlet):
        rho = 1.06 # density of blood
        with tf.GradientTape(watch_accessed_variables=False, persistent=False) as tape:
          #create_tape_time = time.time()
          tape.watch(Q_tensor) # track operations applied to Q_tensor
          U_tensor = tf.divide(Q_tensor, areas_tensor)
          C_outlets, outlets = junction_loss_coeff_tf(U_tensor, areas_tensor, angles_tensor) # run Unified0D junction loss coefficient function
          dP_unified0d_outlets = (rho*tf.multiply(
              C_outlets, tf.square(tf.boolean_mask(U_tensor, outlets))))
          # dP_unified0d_outlets = (rho*tf.multiply(
          #     C_outlets, tf.square(tf.boolean_mask(U_tensor, outlets))) + 0.5*rho*tf.subtract(
          #     tf.square(tf.slice(U_tensor, begin = [max_inlet], size = [1])), tf.square(tf.boolean_mask(U_tensor, outlets)))) # compute pressure loss according to the unified 0d model
        ddP_dU_outlets = tape.jacobian(dP_unified0d_outlets, Q_tensor) # get derivatives of pressure loss coeff wrt. U_tensor
        return dP_unified0d_outlets, ddP_dU_outlets, outlets

    def apply_unified0d(self, args, Q, areas, angles):
        inlet_indices, max_inlet, num_inlets, outlet_indices, num_outlets, num_branches = self.classify_branches(Q)
        angles = self.configure_angles(copy.deepcopy(angles), Q, max_inlet, num_branches) # configure angles for unified0d
        Q_tensor = tf.Variable(tf.convert_to_tensor(Q)) # convert Q to a tensor
        areas_tensor = tf.convert_to_tensor(areas, dtype="double") # convert areas to a tensor
        angles_tensor = tf.convert_to_tensor(angles, dtype="double") # convert angles to a tensor
        outlet_indices_tensor = tf.convert_to_tensor(outlet_indices, dtype="double") # convert U to a tensor
        max_inlet_int = tf.constant(max_inlet, dtype="int32") # convert max_inlet to a constant
        num_outlets_int = tf.constant(num_outlets, dtype="int32") # convert num_outlets to a constant
        start_time = time.time()
        branch_config = tuple(inlet_indices + outlet_indices)
        if branch_config in args["tf_graph_dict"]:
          unified_tf_concrete = args["tf_graph_dict"][branch_config]
        else:
          unified_tf_concrete = self.unified0d_tf.get_concrete_function(Q_tensor, areas_tensor, angles_tensor, num_outlets_int, outlet_indices_tensor, max_inlet_int)
          args["tf_graph_dict"].update({branch_config: unified_tf_concrete})
          print("adding tf.concrete_function for ", branch_config)
        dP_unified0d_outlets, ddP_dU_outlets, outlets = unified_tf_concrete(Q_tensor, areas_tensor, angles_tensor, num_outlets_int, outlet_indices_tensor, max_inlet_int)

        end_time = time.time()
        #print(U)
        if end_time - start_time > 0.1: print("time to apply unified0d: ", end_time - start_time)
        assert tf.size(dP_unified0d_outlets) == num_outlets, "One coefficient should be returned per outlet."
        assert np.array_equal(Q[outlets.numpy()], Q[outlet_indices])
        return dP_unified0d_outlets.numpy(), ddP_dU_outlets.numpy(), outlets

    def set_C(self, dP_unified0d_outlets, Q, areas):
        inlet_indices, max_inlet, num_inlets, outlet_indices, num_outlets, num_branches = self.classify_branches(Q)
        C_vec = [0] * num_branches
        for i in range(num_branches):
          if i in outlet_indices:
            C_vec[i] =  -1*dP_unified0d_outlets[outlet_indices.index(i)]
          # elif i != max_inlet:
          #   C_vec[i] = - 0.5 * self.rho * (np.square(Q[max_inlet]/areas[max_inlet]) - np.square(Q[i]/areas[i]))
        C_vec.pop(max_inlet) # remove dominant inlet entry
        C_vec = C_vec + [0] # mass conservation
        self.mat["C"] = C_vec # set c vector

    def set_dC(self, ddP_dU_outlets, Q, Q_freeze_ind, areas):
        inlet_indices, max_inlet, num_inlets, outlet_indices, num_outlets, num_branches = self.classify_branches(Q)

        for small_Q_ind in Q_freeze_ind:
            ddP_dU_outlets[:, small_Q_ind] = 0

        dC_sign = [-1*self.flow_directions[i] for i in range(len(self.flow_directions))]
        dC = []
        for i in range(num_branches+1):
            dP_derivs = [0] * (2*num_branches)
            if i == max_inlet:
                continue
            elif i in outlet_indices: # loop over pressure losses (rows of C)
                ddP_dU_vec = ddP_dU_outlets[outlet_indices.index(i),:]
                for j in range(num_branches): # loop over velocity derivatives
                    dP_derivs[2*j + 1] = -1*ddP_dU_vec[j] * dC_sign[j]
            # elif i in inlet_indices:
            #     dP_derivs[2*max_inlet + 1] = - self.rho * (Q[max_inlet]/areas[max_inlet])* (1/areas[max_inlet]) * dC_sign[max_inlet]
            #     dP_derivs[2*i + 1] = self.rho * (Q[i]/areas[i])* (1/areas[i]) * dC_sign[i]
            dC.append(tuple(dP_derivs))
        self.mat["dC"] = dC # set c vector

    def update_solution(self, args):
        curr_y, wire_dict, Q, areas, angles, small_Q = self.unpack_params(args)
        Q_frozen, Q_freeze_ind = self.check_rep_Q(Q)
        if np.sum(np.asarray(Q)!=0) == 0: # if all velocities are 0, treat as normal junction (copy-pasted from normal junction code)
            self.set_no_flow_mats() # set F matrix and c vector for zero flow case
            print("all zeros")
        else: # otherwise apply Unified0D model
            self.set_F(Q) # set F matrix
            dP_outlets, ddP_dU_outlets, outlets = self.apply_unified0d(args, Q_frozen, areas, copy.deepcopy(angles)) # apply unified0d
            self.set_C(dP_outlets, Q_frozen, areas) # set C vector
            self.set_dC(ddP_dU_outlets, Q_frozen, Q_freeze_ind, areas) # set dC matrix
            #pdb.set_trace()

class DNNJunction(Junction):

    def __init__(self, junction_parameters, connecting_block_list=None, name="NoNameJunction", flow_directions=None):
        Junction.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.flow_directions = flow_directions
        self.junction_parameters = junction_parameters

    def unpack_params(self, args):
        self.model = args["dnn_model"]["model"]
        self.mu_in = tf.constant(args["dnn_model"]["scalings"]["input_mean"], dtype= "float64")
        self.sd_in = tf.constant(args["dnn_model"]["scalings"]["input_sd"], dtype= "float64")
        self.mu_out = tf.constant(args["dnn_model"]["scalings"]["output_mean"], dtype= "float64")
        self.sd_out = tf.constant(args["dnn_model"]["scalings"]["output_sd"], dtype= "float64")
        curr_y = args['Solution']  # the current solution for all unknowns in our 0D model
        wire_dict = args['Wire dictionary'] # connectivity dictionary
        areas = self.junction_parameters["areas"] # load areas
        Q = np.asarray([-1*self.flow_directions[i] * curr_y[wire_dict[self.connecting_wires_list[i]].LPN_solution_ids[1]] for i in range(len(self.flow_directions))]) # calculate velocity in each branch (inlets positive, outlets negative)
        angles = copy.deepcopy(self.junction_parameters["angles"]) # load angles
        angles.insert(0,0) # add in the angle for the input file "presumed inlet" (first entry)
        return curr_y, wire_dict, Q, areas, angles

    def F_add_continuity_row(self):
        tmp = (0,)
        for d in self.flow_directions[:-1]:
            tmp += (d,)
            tmp += (0,)
        tmp += (self.flow_directions[-1],)
        self.mat['F'].append(tmp)

    def classify_branches(self, U):
        inlet_indices = list(np.asarray(np.nonzero(U>0)).astype(int)[0]) # identify inlets
        U_in = U[inlet_indices]
        max_inlet_tol = 0.01
        max_inlet_cands = inlet_indices[U_in > (np.argmax(U_in)-max_inlet_tol)]
        max_inlet = np.argmin(max_inlet_cands)
        #max_inlet = np.argmax(U) # index of the inlet with max velocity (serves as dominant inlet where necessary)
        num_inlets = len(inlet_indices) # number of inlets
        outlet_indices = list(np.nonzero(U<=0)[0]) # identify outlets
        num_outlets = len(outlet_indices) # number of outlets
        num_branches = U.size # number of branches
        # inlet/outlet sanity checks
        if num_inlets ==0:
          pdb.set_trace()
        assert num_inlets != 0, "No junction inlet."
        assert num_outlets !=0, "No junction outlet."
        assert num_inlets + num_outlets == num_branches, "Sum of inlets and outlets does not equal the number of branches."
        return inlet_indices, max_inlet, num_inlets, outlet_indices, num_outlets, num_branches

    def set_no_flow_mats(self):
        self.mat['F'] = [(1.,) + (0,) * (2 * i + 1) + (-1,) + (0,) * (2 * self.num_connections - 2 * i - 3) for i in
                         range(self.num_connections - 1)] # copy-pasted from normal junction
        self.F_add_continuity_row() # add mass conservation row

    def set_F(self, U):
        inlet_indices, max_inlet, num_inlets, outlet_indices, num_outlets, num_branches = self.classify_branches(U)
        self.mat['F'] = []
        for i in range(0,num_branches): # loop over branches- each branch (exept the "presumed inlet" is a row of F)
            if i == max_inlet:
              continue # if the branch is the dominant inlet branch do not add a new column
            F_row = [0]*(2*num_branches) # row of 0s with 1 in column corresponding to "presumed inlet" branch pressure
            F_row[2*max_inlet] = 1 # place 1 in column corresponding to dominant inlet pressure
            F_row[2*i] = -1 # place -1 in column corresponding to each branch pressure
            self.mat['F'].append(tuple(F_row)) # append row to F matrix
        self.F_add_continuity_row() # add mass conservation row

    @tf.function
    def dnn_tf(self, dnn_input_tensor):
        with tf.GradientTape(watch_accessed_variables=False, persistent=False) as tape:
          tape.watch(dnn_input_tensor) # track operations applied to DNN_input_tensor
          dnn_input_tensor_norm = (dnn_input_tensor-self.mu_in)/self.sd_in
          dnn_output_norm = tf.cast(self.model(dnn_input_tensor_norm), dtype = "float64")
          dP_dnn_outlet = (dnn_output_norm * self.sd_out) + self.mu_out
        ddP_dU_outlet = tape.jacobian(dP_dnn_outlet, dnn_input_tensor) # get derivatives of pressure loss coeff wrt. U_tensor
        return dP_dnn_outlet, ddP_dU_outlet

    def get_dnn_input(self, args, max_inlet, outlet_index, Q, areas, angles):
      curr_y = args['Solution']  # the current solution for all unknowns in our 0D model
      wire_dict = args['Wire dictionary'] # connectivity dictionary
      inlet_indices, max_inlet, num_inlets, outlet_indices, num_outlets, num_branches = self.classify_branches(Q)
      dnn_input = [curr_y[wire_dict[self.connecting_wires_list[max_inlet]].LPN_solution_ids[0]], # inlet pressure
                Q[max_inlet]/areas[max_inlet], # inlet velocity
                -Q[outlet_index]/areas[outlet_index], # outlet velocity,
                Q[max_inlet], # inlet flow
                -Q[outlet_index], # outlet flow
                areas[max_inlet], #inlet area
                areas[outlet_index], # outlet area
                angles[outlet_index], # direction change
                num_outlets, # number of outlets
                num_inlets, # number of inlets
                args["ydot"]] # flow time derivative
      dnn_input_tensor = tf.reshape(tf.convert_to_tensor(dnn_input),[1,11])
      return dnn_input_tensor

    def apply_dnn(self, args, Q, areas, angles, curr_y, wire_dict):
        inlet_indices, max_inlet, num_inlets, outlet_indices, num_outlets, num_branches = self.classify_branches(Q)
        start_time = time.time()
        dP_dnn_outlets = []
        ddP_outlets = []
        for outlet_index in outlet_indices:
            dnn_input_tensor = self.get_dnn_input(args, max_inlet, outlet_index, Q, areas, angles)
            if "dnn_graph" in args["tf_graph_dict"]:
              dnn_tf_concrete = args["tf_graph_dict"]["dnn_graph"]
            else:
              dnn_tf_concrete = self.dnn_tf.get_concrete_function(dnn_input_tensor)
              args["tf_graph_dict"].update({"dnn_graph": dnn_tf_concrete})
              print("adding tf.concrete_function")
            dP_dnn_outlet, ddP_dU_outlet = dnn_tf_concrete(dnn_input_tensor)
            dP_dnn_outlets.append(dP_dnn_outlet)
            ddP_outlets.append(ddP_dU_outlet)
        end_time = time.time()
        #print("delta Ps: " , dP_dnn_outlet, ddP_dU_outlet)
        #pdb.set_trace()
        if end_time - start_time > 0.01: print("time to apply unified0d: ", end_time - start_time)
        assert len(dP_dnn_outlets) == num_outlets, "One coefficient should be returned per outlet."
        return np.concatenate(np.concatenate(np.asarray(dP_dnn_outlets))), np.asarray(ddP_outlets)

    def set_C(self, dP_dnn_outlets, U):
        inlet_indices, max_inlet, num_inlets, outlet_indices, num_outlets, num_branches = self.classify_branches(U)
        C_vec = [0] * num_branches
        for i in range(num_branches):
          if i in outlet_indices:
            #pdb.set_trace()
            C_vec[i] =  -1*dP_dnn_outlets[outlet_indices.index(i)]
        C_vec.pop(max_inlet) # remove dominant inlet entry
        C_vec = C_vec + [0]
        self.mat["C"] = C_vec # set c vector

    def set_dC(self, ddP_outlets, Q, areas):
        inlet_indices, max_inlet, num_inlets, outlet_indices, num_outlets, num_branches = self.classify_branches(Q)
        #pdb.set_trace()
        Q_sign = [-1*self.flow_directions[i] for i in range(len(self.flow_directions))]
        dC = []
        for i in range(num_branches+1):
            dP_derivs = [0] * (2*num_branches)
            if i == max_inlet:
              continue
            elif i in outlet_indices:
              outlet_number = outlet_indices.index(i) # loop over pressure losses (rows of C)
              #pdb.set_trace()
              #print("max_inlet", max_inlet)
              #print("Q", Q)
              dP_derivs[2*max_inlet] = -ddP_outlets[outlet_number][0][0][0][0] # inlet pressure derivative
              dP_derivs[2*max_inlet +1 ] = -1*Q_sign[max_inlet]*(ddP_outlets[outlet_number][0][0][0][1]/areas[max_inlet] + ddP_outlets[outlet_number][0][0][0][3])# inlet_velocity derivative
              dP_derivs[2*i + 1] = 1*Q_sign[i]*(ddP_outlets[outlet_number][0][0][0][2]/areas[i]+ ddP_outlets[outlet_number][0][0][0][4])# outlet_velocity derivative
            dC.append(tuple(dP_derivs))

        self.mat["dC"] = dC # set c vector

    def update_solution(self, args):
        curr_y, wire_dict, U, areas, angles = self.unpack_params(args)
        if np.sum(np.asarray(U)!=0) == 0: # if all velocities are 0, treat as normal junction (copy-pasted from normal junction code)
            self.set_no_flow_mats() # set F matrix and c vector for zero flow case

        else: # otherwise apply Unified0D model
            self.set_F(U) # set F matrix
            dP_outlets, ddP_dU_outlets = self.apply_dnn(args, U, areas, angles, curr_y, wire_dict) # apply unified0d
            self.set_C(dP_outlets, U) # set C vector
            self.set_dC(ddP_dU_outlets, U, areas) # set dC matrix
            #pdb.set_trace()
class BloodVessel(LPNBlock):
    """
    Stenosis:
        equation: delta_P = ( K_t * rho / ( 2 * (A_0)**2 ) ) * ( ( A_0 / A_s ) - 1 )**2 * Q * abs(Q) + R_poiseuille * Q
                          =               stenosis_coefficient                          * Q * abs(Q) + R_poiseuille * Q

        source: Mirramezani, M., Shadden, S.C. A distributed lumped parameter model of blood flow. Annals of Biomedical Engineering. 2020.
    """
    def __init__(self, R, C, L, stenosis_coefficient, connecting_block_list = None, name = "NoNameBloodVessel", flow_directions = None):
        LPNBlock.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.type = "BloodVessel"
        self.R = R  # poiseuille resistance value = 8 * mu * L / (pi * r**4)
        self.C = C
        self.L = L
        self.stenosis_coefficient = stenosis_coefficient

    # the ordering of the solution variables is : (P_in, Q_in, P_out, Q_out)

    def update_constant(self):
        self.mat['E'] = [(0, 0, 0, -self.L), (-self.C, self.C * self.R, 0, 0)]

    def update_solution(self, args):
        curr_y = args['Solution']  # the current solution for all unknowns in our 0D model
        wire_dict = args['Wire dictionary']
        Q_in = curr_y[wire_dict[self.connecting_wires_list[0]].LPN_solution_ids[1]]
        self.mat['F'] = [(1.0, -1.0 * self.stenosis_coefficient * np.abs(Q_in) - self.R, -1.0, 0), (0, 1.0, 0, -1.0)]
        self.mat['dF'] = [(0, -1.0 * self.stenosis_coefficient * np.abs(Q_in), 0, 0), (0,) * 4]

class UnsteadyResistanceWithDistalPressure(LPNBlock):
    def __init__(self, Rfunc, Pref_func, connecting_block_list=None, name="NoNameUnsteadyResistanceWithDistalPressure",
                 flow_directions=None):
        LPNBlock.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.type = "UnsteadyResistanceWithDistalPressure"
        self.neq = 1
        self.Rfunc = Rfunc
        self.Pref_func = Pref_func

    def update_time(self, args):
        """
        the ordering is : (P_in,Q_in)
        """
        t = args['Time']
        self.mat['F'] = [(1., -1.0 * self.Rfunc(t))]
        self.mat['C'] = [-1.0 * self.Pref_func(t)]

class UnsteadyPressureRef(LPNBlock):
    """
    Unsteady P reference
    """
    def __init__(self, Pfunc, connecting_block_list=None, name="NoNameUnsteadyPressureRef", flow_directions=None):
        LPNBlock.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.type = "UnsteadyPressureRef"
        self.neq = 1
        self.n_connect = 1
        self.Pfunc = Pfunc

    def update_time(self, args):
        t = args['Time']
        self.mat['C'] = [-1.0 * self.Pfunc(t)]

    def update_constant(self):
        self.mat['F'] = [(1., 0.)]

class UnsteadyFlowRef(LPNBlock):
    """
    Flow reference
    """
    def __init__(self, Qfunc, connecting_block_list=None, name="NoNameUnsteadyFlowRef", flow_directions=None):
        LPNBlock.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.type = "UnsteadyFlowRef"
        self.neq = 1
        self.n_connect = 1
        self.Qfunc = Qfunc

    def update_time(self, args):
        t = args['Time']
        self.mat['C'] = [-1.0 * self.Qfunc(t)]

    def update_constant(self):
        self.mat['F'] = [(0, 1.)]

class UnsteadyRCRBlockWithDistalPressure(LPNBlock):
    """
    Unsteady RCR - time-varying RCR values
    Formulation includes additional variable : internal pressure proximal to capacitance.
    """
    def __init__(self, Rp_func, C_func, Rd_func, Pref_func, connecting_block_list=None,
                 name="NoNameUnsteadyRCRBlockWithDistalPressure", flow_directions=None):
        LPNBlock.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.type = "UnsteadyRCRBlockWithDistalPressure"
        self.neq = 2
        self.n_connect = 1
        self.num_block_vars = 1
        self.Rp_func = Rp_func
        self.C_func = C_func
        self.Rd_func = Rd_func
        self.Pref_func = Pref_func

    def update_time(self, args):
        """
        unknowns = [P_in, Q_in, internal_var (Pressure at the intersection of the Rp, Rd, and C elements)]
        """
        t = args['Time']
        self.mat['E'] = [(0, 0, 0), (0, 0, -1.0 * self.Rd_func(t) * self.C_func(t))]
        self.mat['F'] = [(1., -self.Rp_func(t), -1.), (0.0, self.Rd_func(t), -1.0)]
        self.mat['C'] = [0, self.Pref_func(t)]

class OpenLoopCoronaryWithDistalPressureBlock(LPNBlock):
    """
    open-loop coronary BC = RCRCR BC
    Publication reference: Kim, H. J. et al. Patient-specific modeling of blood flow and pressure in human coronary arteries. Annals of Biomedical Engineering 38, 3195–3209 (2010)."
    """
    def __init__(self, Ra, Ca, Ram, Cim, Rv, Pim, Pv, cardiac_cycle_period, connecting_block_list=None,
                 name="NoNameCoronary", flow_directions=None):
        LPNBlock.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.type = "OpenLoopCoronaryWithDistalPressureBlock_v2"
        self.neq = 2
        self.n_connect = 1
        self.num_block_vars = 1
        self.Ra = Ra
        self.Ca = Ca
        self.Ram = Ram
        self.Cim = Cim
        self.Rv = Rv
        self.Pa = 0.0
        self.Pim = Pim
        self.Pv = Pv
        self.cardiac_cycle_period = cardiac_cycle_period

    def get_P_at_t(self, P, t):
        tt = P[:, 0]
        P_val = P[:, 1]
        ti, td = divmod(t, self.cardiac_cycle_period)
        P_tt = np.interp(td, tt, P_val)
        return P_tt

    def update_time(self, args):
        # For this open-loop coronary BC, the ordering of solution unknowns is : (P_in, Q_in, V_im)
        # where V_im is the volume of the second capacitor, Cim
        # Q_in is the flow through the first resistor
        # and P_in is the pressure at the inlet of the first resistor
        ttt = args['Time']
        Pim_value = self.get_P_at_t(self.Pim, ttt)
        Pv_value = self.get_P_at_t(self.Pv, ttt)
        self.mat['C'] = [-1.0 * self.Cim * Pim_value + self.Cim * Pv_value,
                         -1.0 * self.Cim * (self.Rv + self.Ram) * Pim_value + self.Ram * self.Cim * Pv_value]

    def update_constant(self):
        self.mat['E'] = [
            (-1.0 * self.Ca * self.Cim * self.Rv, self.Ra * self.Ca * self.Cim * self.Rv, -1.0 * self.Cim * self.Rv),
            (0.0, 0.0, -1.0 * self.Cim * self.Rv * self.Ram)]
        self.mat['F'] = [(0.0, self.Cim * self.Rv, -1.0),
                         (self.Cim * self.Rv, -1.0 * self.Cim * self.Rv * self.Ra, -1.0 * (self.Rv + self.Ram))]
