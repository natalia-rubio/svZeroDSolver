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
import tensorflow as tf

class wire:
    """
    Wires connect circuit elements and junctions
    They can only posses a single pressure and flow value (system variables)
    They can also only possess one element(or junction) at each end
    """
    def __init__(self, connecting_elements, name="NoNameWire", P_units='cgs', Q_units='cgs'):
        self.name = name
        self.type = 'Wire'
        if len(connecting_elements) > 2:
            raise Exception('Wire cannot connect to more than two elements at a time. Use a junction LPN block')
        if not isinstance(connecting_elements, tuple):
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
        self.type = "ArbitraryBlock"
        self.num_block_vars = 0
        self.connecting_wires_list = []

        # -1 : Inflow to block, +1 outflow from block
        self.flow_directions = flow_directions

        # solution IDs for the LPN block's internal solution variables
        self.LPN_solution_ids = []

        # block matrices
        self.mat = {}
        self.vec = {}

        # mat and vec assembly queue. To reduce the need to reassemble
        # matrices that havent't changed since last assembly, these attributes
        # are used to queue updated mats and vecs.
        self.mats_to_assemble = set()
        self.vecs_to_assemble = set()

        # row and column indices of block in global matrix
        self.global_col_id = []
        self.global_row_id = []

        self.flat_row_ids = []
        self.flat_col_ids = []

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

class InternalJunction(LPNBlock):
    """
    Internal junction points between LPN blocks (for mesh refinement, does not appear as physical junction in model)
    """
    def __init__(self, connecting_block_list=None, name="NoNameJunction", flow_directions=None):
        LPNBlock.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.type = "Junction"
        self.neq = self.num_connections  # number of equations = num of blocks that connect to this junction, where the equations are 1) mass conservation 2) inlet pressures = outlet pressures

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
        self.mat['F'] = np.array(self.mat['F'], dtype=float)
        self.mats_to_assemble.add("F")

    def add_connecting_block(self, block, direction):
        self.connecting_block_list.append(block)
        self.num_connections = len(self.connecting_block_list)
        self.neq = self.num_connections
        self.flow_directions.append(direction)

class BloodVesselJunction(InternalJunction):
    """
    Blood vessel junction (dummy for future implementation of blood pressure losses at junctions)
    """
    def __init__(self, j_params, connecting_block_list=None, name="NoNameJunction", flow_directions=None):
        InternalJunction.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.j_params = j_params

class TotalPressureJunction(InternalJunction):
    """
    Blood vessel junction (dummy for future implementation of blood pressure losses at junctions)
    """
    def __init__(self, j_params, connecting_block_list=None, name="NoNameJunction", flow_directions=None):
        InternalJunction.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.j_params = j_params
        self.rho = 1.06

    def F_add_continuity_row(self):
        tmp = (0,)
        for d in self.flow_directions[:-1]:
            tmp += (d,)
            tmp += (0,)
        tmp += (self.flow_directions[-1],)
        self.mat['F'].append(tmp)

    def unpack_params(self, args):
        curr_y = args['Solution']  # the current solution for all unknowns in our 0D model
        wire_dict = args['Wire dictionary'] # connectivity dictionary
        areas = self.j_params["areas"] # load areas
        Q = np.asarray([-1*self.flow_directions[i] *
            curr_y[wire_dict[self.connecting_wires_list[i]].LPN_solution_ids[1]] for i in range(len(self.flow_directions))]) # calculate velocity in each branch (inlets positive, outlets negative)
        V = np.divide(Q,areas)
        max_inlet  = np.argmax(Q)
        return curr_y, wire_dict, Q, areas, max_inlet

    def set_F(self, Q, max_inlet):
        self.mat['F'] = []
        num_branches = len(Q)
        for i in range(0,num_branches): # loop over branches- each branch (exept the "presumed inlet" is a row of F)
            if i == max_inlet:
              continue # if the branch is the dominant inlet branch do not add a new row
            F_row = [0]*(2*num_branches) # row of 0s with 1 in column corresponding to "presumed inlet" branch pressure
            F_row[2*max_inlet] = 1 # place 1 in column corresponding to dominant inlet pressure
            F_row[2*i] = -1 # place -1 in column corresponding to each branch pressure
            self.mat['F'].append(tuple(F_row)) # append row to F matrix
        self.F_add_continuity_row() # add mass conservation row
        self.mat['F'] = np.array(self.mat['F'])
        self.mats_to_assemble.add("F")

    @tf.function
    def static_p_tf(self, Q_tf, areas_tf, max_inlet):
        rho = tf.constant(1.06, dtype=tf.float32) # density of blood
        with tf.GradientTape(watch_accessed_variables=False, persistent=False) as tape:
          tape.watch(Q_tf) # track operations applied to Q_tensor
          V = tf.math.abs(tf.math.divide(Q_tf,areas_tf))
          C = tf.divide(tf.multiply(rho, tf.multiply(tf.constant(0.5, dtype=tf.float32), tf.subtract(
           tf.math.square(tf.slice(V, begin = [max_inlet], size = [1])), tf.math.square(V)))),
           tf.constant(1333, dtype=tf.float32))

        dC_dQ = tape.jacobian(C, Q_tf) # get derivatives of pressure loss coeff wrt. U_tensor
        return C, dC_dQ

    def set_C(self, args, Q, areas, max_inlet):

        Q_tf = tf.Variable(tf.convert_to_tensor(Q, dtype=tf.float32)) # convert Q to a tensor
        areas_tf = tf.convert_to_tensor(areas, dtype=tf.float32) # convert areas to a tensor
        branch_config = (Q.size, max_inlet)
        max_inlet = tf.convert_to_tensor(max_inlet, dtype=tf.int32)
        if branch_config in args["tf_graph_dict"]:
          unified_tf_concrete = args["tf_graph_dict"][branch_config]
        else:
          unified_tf_concrete = self.static_p_tf.get_concrete_function(Q_tf, areas_tf, max_inlet)
          args["tf_graph_dict"].update({branch_config: unified_tf_concrete})

        C_tf, dC_dQ_tf = unified_tf_concrete(Q_tf, areas_tf, max_inlet)
        C_list = C_tf.numpy().tolist(); C_list.pop(max_inlet)
        self.vec["C"] = np.array(C_list + [0])
        self.vecs_to_assemble.add("C")

        dC_dQ_np = dC_dQ_tf.numpy()

        dC_dsol = []
        for i in range(len(Q)):
            if i == max_inlet:  continue
            deriv_list = []
            for j in range(len(Q)): # loop over velocity derivatives
                deriv_list.append(0); deriv_list.append(-1*self.flow_directions[i]*dC_dQ_np[i,j])
            dC_dsol.append(tuple(deriv_list))
        dC_dsol.append(tuple([0*i for i in deriv_list]))
        self.mat["dC"] = np.array(dC_dsol)
        self.mats_to_assemble.add("dC")
        return

    def update_solution(self, args):
        curr_y, wire_dict, Q, areas, max_inlet = self.unpack_params(args)

        if np.sum(np.asarray(Q)!=0) == 0: # if all velocities are 0, treat as normal junction (copy-pasted from normal junction code)
            self.set_F(Q, max_inlet) # set F matrix and c vector for zero flow case
            print("all zeros")
        else: # otherwise apply Unified0D model
            self.set_F(Q, max_inlet) # set F matrix
            self.set_C(args, Q, areas, max_inlet)
            # print(f"Q: {Q}")

class Unified0DJunction(InternalJunction):
    """
    Blood vessel junction (dummy for future implementation of blood pressure losses at junctions)
    """
    def __init__(self, j_params, connecting_block_list=None, name="NoNameJunction", flow_directions=None):
        InternalJunction.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.j_params = j_params
        self.rho = 1.06

    def F_add_continuity_row(self):
        tmp = (0,)
        for d in self.flow_directions[:-1]:
            tmp += (d,)
            tmp += (0,)
        tmp += (self.flow_directions[-1],)
        self.mat['F'].append(tmp)

    def unpack_params(self, args):
        curr_y = args['Solution']  # the current solution for all unknowns in 0D model
        wire_dict = args['Wire dictionary'] # connectivity dictionary
        areas = self.j_params["areas"] # load areas
        Q = np.asarray([-1*self.flow_directions[i] *
            curr_y[wire_dict[self.connecting_wires_list[i]].LPN_solution_ids[1]] for i in range(len(self.flow_directions))]) # flow in each branch (inlets +, outlets -)
        angles = copy.deepcopy(self.j_params["angles"]) # load angles
        return curr_y, wire_dict, Q, areas, angles

    def classify_branches(self, Q):
        inlet_indices = list(np.asarray(np.nonzero(Q>=0)).astype(int)[0]) # identify inlets
        Q_in = Q[inlet_indices]
        max_inlet_tol = 0.01
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
        self.mats_to_assemble.add("F")

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
        self.mats_to_assemble.add("F")

    def configure_angles(self, angles, max_inlet, num_branches):
        angle_shift = np.pi - angles[max_inlet] # find shift to set first inlet angle to pi
        for i in range(num_branches): # loop over all angles
            angles[i] = angles[i] + angle_shift # shift all junction angles
        assert len(angles) == num_branches, 'One angle should be provided for each branch'
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
              C_outlets, tf.square(tf.boolean_mask(U_tensor, outlets))) + 0.5*rho*tf.subtract(
              tf.square(tf.slice(U_tensor, begin = [max_inlet], size = [1])), tf.square(tf.boolean_mask(U_tensor, outlets)))) # compute pressure loss according to the unified 0d model
        ddP_dU_outlets = tape.jacobian(dP_unified0d_outlets, Q_tensor) # get derivatives of pressure loss coeff wrt. U_tensor
        return dP_unified0d_outlets, ddP_dU_outlets, outlets

    def apply_unified0d(self, args, Q, areas, angles):
        inlet_indices, max_inlet, num_inlets, outlet_indices, num_outlets, num_branches = self.classify_branches(Q)
        angles = self.configure_angles(copy.deepcopy(angles), max_inlet, num_branches) # configure angles for unified0d
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
          elif i != max_inlet:
            C_vec[i] = - 0.5 * self.rho * (np.square(Q[max_inlet]/areas[max_inlet]) - np.square(Q[i]/areas[i]))
        C_vec.pop(max_inlet) # remove dominant inlet entry
        C_vec = C_vec + [0] # mass conservation
        self.mat["C"] = C_vec # set C vector
        self.vecs_to_assemble.add("C")

        dC_dQ_np = dC_dQ_tf.numpy()

        dC_dsol = []
        for i in range(len(Q)):
            if i == max_inlet:  continue
            deriv_list = []
            for j in range(len(Q)): # loop over velocity derivatives
                deriv_list.append(0); deriv_list.append(-1*self.flow_directions[i]*dC_dQ_np[i,j])
            dC_dsol.append(tuple(deriv_list))
        dC_dsol.append(tuple([0*i for i in deriv_list]))
        self.mat["dC"] = dC_dsol
        self.mats_to_assemble.add["dC"]
        return

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
        self.neq = 3
        self.num_block_vars = 1
        self.R = R  # poiseuille resistance value = 8 * mu * L / (pi * r**4)
        self.C = C
        self.L = L
        self.stenosis_coefficient = stenosis_coefficient
        self._qin_id = None

        # the ordering of the solution variables is : (P_in, Q_in, P_out, Q_out)
        self.mat['E'] = np.zeros((3, 5), dtype=float)
        self.mat["E"][0, 3] = -self.L
        self.mat['E'][1, 4] = -self.C
        self.mat['F'] = np.array(
            [
                [1.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, -1.0]
            ],
            dtype=float
        )
        self.mat['dF'] = np.zeros((3, 5), dtype=float)

        # only necessary to assemble E in __init__, F and dF get assembled with update_solution
        self.mats_to_assemble.add("E")

    def update_solution(self, args):
        Q_in = np.abs(args["Solution"][args['Wire dictionary'][self.connecting_wires_list[0]].LPN_solution_ids[1]])
        fac1 = -self.stenosis_coefficient * Q_in
        fac2 = fac1 - self.R
        self.mat['F'][[0, 2], 1] = fac2
        self.mat['dF'][[0, 2], 1] = fac1
        self.mats_to_assemble.update({"F", "dF"})

class UnsteadyResistanceWithDistalPressure(LPNBlock):
    def __init__(self, Rfunc, Pref_func, connecting_block_list=None, name="NoNameUnsteadyResistanceWithDistalPressure",
                 flow_directions=None):
        LPNBlock.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.type = "UnsteadyResistanceWithDistalPressure"
        self.neq = 1
        self.Rfunc = Rfunc
        self.Pref_func = Pref_func
        self.mat["F"] = np.array(
            [
                [1.0, 0.0],
            ],
            dtype=float
        )
        self.vec['C'] = np.array([0.0], dtype=float)

    def update_time(self, args):
        """
        the ordering is : (P_in,Q_in)
        """
        t = args['Time']
        self.mat["F"][0, 1] = -self.Rfunc(t)
        self.vec['C'][0] = -self.Pref_func(t)
        self.mats_to_assemble.add("F")
        self.vecs_to_assemble.add("C")

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

        self.vec["C"] = np.zeros(1, dtype=float)
        self.mat["F"] = np.array([[1.0, 0.0]], dtype=float)

    def update_time(self, args):
        t = args['Time']
        self.vec['C'][0] = -self.Pfunc(t)
        self.vecs_to_assemble.add("C")
        self.mats_to_assemble.add("F")

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
        self.vec['C'] = np.zeros(1, dtype=float)
        self.mat["F"] = np.array([[0.0, 1.0]], dtype=float)

    def update_time(self, args):
        t = args['Time']
        self.vec['C'][0] = -self.Qfunc(t)
        self.vecs_to_assemble.add("C")
        self.mats_to_assemble.add("F")

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

        self.mat["E"] = np.zeros((2, 3), dtype=float)
        self.mat['F'] = np.array(
            [
                [1.0, 0.0, -1.0],
                [0.0, 0.0, -1.0]
            ],
            dtype=float
        )
        self.vec['C'] = np.array([0.0, 0.0], dtype=float)

    def update_time(self, args):
        """
        unknowns = [P_in, Q_in, internal_var (Pressure at the intersection of the Rp, Rd, and C elements)]
        """
        t = args['Time']
        Rd_t = self.Rd_func(t)
        self.mat["E"][1, 2] = -Rd_t * self.C_func(t)
        self.mat['F'][0, 1] = -self.Rp_func(t)
        self.mat['F'][1, 1] = Rd_t
        self.vec['C'][1] = self.Pref_func(t)
        self.mats_to_assemble.update({"E", "F"})
        self.vecs_to_assemble.add("C")

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

        self.vec['C'] = np.zeros(2)
        self.mat['E'] = np.zeros((2, 3))
        self.mat['F'] = np.zeros((2, 3))
        self.mat['F'][0, 2] = -1.0

        Cim_Rv = self.Cim * self.Rv
        self.mat['E'][0, 0] = -self.Ca * Cim_Rv
        self.mat['E'][0, 1] = self.Ra * self.Ca * Cim_Rv
        self.mat['E'][0, 2] = -Cim_Rv
        self.mat['E'][1, 2] = -Cim_Rv * self.Ram
        self.mat['F'][0, 1] = Cim_Rv
        self.mat['F'][1, 0] = Cim_Rv
        self.mat['F'][1, 1] = -Cim_Rv * self.Ra
        self.mat['F'][1, 2] = -(self.Rv + self.Ram)
        self.mats_to_assemble.update({"E", "F"})

    def get_P_at_t(self, P, t):
        tt = P[:, 0]
        P_val = P[:, 1]
        _, td = divmod(t, self.cardiac_cycle_period)
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
        self.vec["C"][0] = -self.Cim * Pim_value + self.Cim * Pv_value
        self.vec["C"][1] = -self.Cim * (self.Rv + self.Ram) * Pim_value + self.Ram * self.Cim * Pv_value
        self.vecs_to_assemble.add("C")
