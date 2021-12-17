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
from .junction_loss_coeff import junction_loss_coeff
import tensorflow as tf
from tensorflow import keras
import pickle
DNN_model = keras.models.load_model('../svzerodsolver/DNN_model_oct')
from sklearn.preprocessing import StandardScaler
z_scaler_in = pickle.load(open('../svzerodsolver/z_scaling/scale_z_in/z_scaler_in.pkl', 'rb')) # scaler between normalized and physical domain
z_scaler_out = pickle.load(open('../svzerodsolver/z_scaling/scale_z_out/z_scaler_out.pkl', 'rb')) # scaler between normalized and physical domain

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

class UNIFIED0DJunction(Junction):
    def __init__(self, junction_parameters, connecting_block_list=None, name="NoNameJunction", flow_directions=None):
        Junction.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions)
        self.flow_directions = flow_directions
        self.junction_parameters = junction_parameters

    def update_solution(self, args):
        #print("updating solution: new Newton iteration")
        curr_y = args['Solution']  # the current solution for all unknowns in our 0D model
        wire_dict = args['Wire dictionary'] # connectivity dictionary
        rho = 1.06 # density of blood
        areas = self.junction_parameters["areas"] # load areas
        U = np.asarray([-1*self.flow_directions[i] * np.divide(
            curr_y[wire_dict[self.connecting_wires_list[i]].LPN_solution_ids[1]],
            areas[i]) for i in range(len(self.flow_directions))]) # calculate velocity in each branch (inlets positive, outlets negative)
        #pdb.set_trace()
        if np.sum(np.asarray(U)!=0) == 0: # if all velociies are 0, treat as normal junction (copy-pasted from normal junction code)
            self.mat['F'] = [(1.,) + (0,) * (2 * i + 1) + (-1,) + (0,) * (2 * self.num_connections - 2 * i - 3) for i in
                         range(self.num_connections - 1)]
            tmp = (0,)
            for d in self.flow_directions[:-1]:
                tmp += (d,)
                tmp += (0,)
            tmp += (self.flow_directions[-1],)
            self.mat['F'].append(tmp)
        else: # otherwise apply Unified0D model
            #print("nonzero velocities")
            # IDENTIFY INLETS AND OUTLETS
            inlet_indices = list(np.asarray(np.nonzero(U>0)).astype(int)[0]) # identify inlets
            max_inlet = np.argmax(U) # index of the inlet with max velocity (serves as dominant inlet where necessary)
            num_inlets = len(inlet_indices) # number of inlets
            outlet_indices = list(np.nonzero(U<=0)[0]) # identify outlets
            num_outlets = len(outlet_indices) # number of outlets
            num_branches = len(self.flow_directions) # number of branches
            # inlet/outlet sanity checks
            if num_inlets ==0:
              pdb.set_trace()
            #assert num_inlets != 0, "No junction inlet."
            assert num_outlets !=0, "No junction outlet."
            assert num_inlets + num_outlets == num_branches, "Sum of inlets and outlets does not equal the number of branches."

            # CONFIGURE ANGLES FOR UNIFIED0D
            angles = copy.deepcopy(self.junction_parameters["angles"]) # load angles
            angles.insert(0,0) # add in the angle for the input file "presumed inlet" (first entry)
            angle_shift = np.pi - angles[max_inlet] # find shift to set first inlet angle to pi
            for i in range(num_branches): # loop over all angles
                angles[i] = angles[i] + angle_shift # shift all junction angles
            assert len(angles) == num_branches, 'One angle should be provided for each branch'


            # SET F MATRIX
            self.mat['F'] = []
            for i in range(0,num_branches): # loop over branches- each branch (exept the "presumed inlet" is a row of F)
                if i == max_inlet:
                  continue # if the branch is the dominant inlet branch do not add a new column
                F_row = [0]*(2*num_branches - 1) # row of 0s with 1 in column corresponding to "presumed inlet" branch pressure
                F_row[2*max_inlet] = 1 # place 1 in column corresponding to dominant inlet pressure
                F_row[2*i] = -1 # place -1 in column corresponding to each branch pressure
                self.mat['F'].append(tuple(F_row)) # append row to F matrix
            # mass conservation row (copy-pasted)
            tmp = (0,)
            for d in self.flow_directions[:-1]:
                tmp += (d,)
                tmp += (0,)
            tmp += (self.flow_directions[-1],) # mass conservation row
            self.mat['F'].append(tmp) # append row to F matrix

            # SET C VECTOR
            C, K, eta_j = junction_loss_coeff(U, np.asarray(areas), np.asarray(angles)) # run Unified0D junction loss coefficient function
            assert np.size(C) == num_branches, "One coefficient should be returned per branch."
            pressure_loss_unified0d = (rho*np.multiply(
                C, np.square(U)) + 0.5*rho*np.subtract(
                np.square(U[max_inlet]), np.square(U))) # compute pressure loss according to the unified 0d model
            self.mat['C']  =  [-1*pressure_loss_unified0d[i] for i in range(num_branches)] + [0] # set C vector
            self.mat['C'].pop(max_inlet)
            # SET dC MATRIX
            Q = np.abs(np.divide(U,areas))
            unified0D_derivs_all = []
            for i in range(0,num_branches+1):
                unified0D_derivs = 2*len(self.flow_directions)*[0,]
                inlet_index = max_inlet
                if i == max_inlet:
                    continue
                if i in outlet_indices:
                    outlet_index = i
                    pdb.set_trace()
                    try:
                        dQ_in = (Q[inlet_index]*rho)/areas[inlet_index]**2 + (5*Q[outlet_index]**3*rho*np.exp((
                            5*Q[outlet_index])/Q[inlet_index])*((areas[outlet_index]*Q[inlet_index]*np.cos((
                            3*angles[outlet_index])/4 + np.pi/4)*(eta_j[outlet_indices.index(outlet_index)] - 1))/(
                            areas[inlet_index]*Q[outlet_index]) + 1))/(areas[outlet_index]**2*Q[inlet_index]**2) - (
                            Q[outlet_index]*rho*np.cos((3*angles[outlet_index])/4 + np.pi/4)*(np.exp((
                            5*Q[outlet_index])/Q[inlet_index]) - 1)*(eta_j[outlet_indices.index(outlet_index)] - 1))/(
                            areas[inlet_index]*areas[outlet_index])

                        dQ_out = (Q[inlet_index]*rho*np.cos((3*angles[outlet_index])/4 + np.pi/4)*(np.exp((
                            5*Q[outlet_index])/Q[inlet_index]) - 1)*(eta_j[outlet_indices.index(outlet_index)] - 1))/(
                            areas[inlet_index]*areas[outlet_index]) - (2*Q[outlet_index]*rho*(np.exp((
                            5*Q[outlet_index])/Q[inlet_index]) - 1)*((areas[outlet_index]*Q[inlet_index]*np.cos((
                            3*angles[outlet_index])/4 + np.pi/4)*(eta_j[outlet_indices.index(
                            outlet_index)] - 1))/(areas[inlet_index]*Q[outlet_index]) + 1))/areas[outlet_index]**2 - (
                            5*Q[outlet_index]**2*rho*np.exp((5*Q[outlet_index])/Q[inlet_index])*((
                            areas[outlet_index]*Q[inlet_index]*np.cos((3*angles[outlet_index])/4 + np.pi/4)*(
                            eta_j[outlet_indices.index(outlet_index)] - 1))/(areas[inlet_index]*Q[outlet_index]) + 1))/(
                            areas[outlet_index]**2*Q[inlet_index]) - (Q[outlet_index]*rho)/areas[outlet_index]**2

                    except:
                        print("error in finding analytical derivative")
                    #pdb.set_trace()

                    unified0D_derivs[2*inlet_index+1] = np.sign(U[inlet_index]) * np.sign(
                        curr_y[wire_dict[self.connecting_wires_list[inlet_index]].LPN_solution_ids[1]]) * dQ_in
                    unified0D_derivs[2*outlet_index+1] = np.sign(U[outlet_index]) * np.sign(
                        curr_y[wire_dict[self.connecting_wires_list[outlet_index]].LPN_solution_ids[1]]) * dQ_out
                    if np.isnan(np.sum(unified0D_derivs))==True:
                        pdb.set_trace()

                unified0D_derivs_all.append(tuple(unified0D_derivs))
            self.mat["dC"] = unified0D_derivs_all
            #pdb.set_trace()

class DNNJunction(Junction):
    def __init__(self, connecting_block_list=None, name="NoNameJunction", flow_directions=None,
                 angles = None, areas = None, lengths = None):
        Junction.__init__(self, connecting_block_list, name=name, flow_directions=flow_directions,angles = angles, areas = areas, lengths = lengths)

        if np.sum(np.asarray(self.flow_directions)<0) != 1 :
            raise ValueError("Junction must have exactly one inlet.")
        self.inlet_index = np.asarray(np.nonzero(np.asarray(flow_directions)<0)).astype(int)[0][0]
        self.outlet_indices = list(np.nonzero(np.asarray(self.flow_directions)>=0)[0])
        self.num_inlets = 1
        self.num_outlets = len(flow_directions)-1
        self.angles = angles.insert(self.inlet_index, np.NaN)
        areass = areas.insert(self.inlet_index, np.NaN)
        self.lengths = lengths.insert(self.inlet_index, np.NaN)
        self.flow_time_der = [bb * 0 + 0.002 for bb in areass]
        # SET GEOMETREIC PARAMETERS
        #areas = [0.5, 0.5, 0.5]
        #self.angles = [0.9,0.9,0.9]
        #self.junction_length = [11,11,11]
        #self.flow_time_der = [0.001, 0.001,0.004]
        self.flow_directions = flow_directions

    def update_solution(self,args):
        inlet_index = self.inlet_index
        outlet_indices = self.outlet_indices
        curr_y = args['Solution']  # the current solution for all unknowns in our 0D model
        wire_dict = args['Wire dictionary']

        # SET F MATRIX
        self.mat['F'] = [(1.,) + (0,) * (2 * i + 1) + (-1,) + (0,) * (2 * self.num_connections - 2 * i - 3) for i in
                 range(self.num_connections - 1)] # pressure drop over junctions rows
        tmp = (0,)
        for d in self.flow_directions[:-1]:
            tmp += (d,)
            tmp += (0,)
        tmp += (self.flow_directions[-1],) # mass conservation
        self.mat['F'].append(tmp)

        # SET C VECTOR
        DNN_input = len(self.flow_directions)*[np.NaN]
        for outlet_index in outlet_indices:
            DNN_input[outlet_index] = np.asarray([curr_y[wire_dict[self.connecting_wires_list[inlet_index]].LPN_solution_ids[0]]/1333.2, # inlet pressure
                curr_y[wire_dict[self.connecting_wires_list[inlet_index]].LPN_solution_ids[1]]/areas[inlet_index], # inlet velocity
                curr_y[wire_dict[self.connecting_wires_list[outlet_index]].LPN_solution_ids[1]]/areas[outlet_index], # outlet velocity,
                curr_y[wire_dict[self.connecting_wires_list[inlet_index]].LPN_solution_ids[1]], # inlet flow
                curr_y[wire_dict[self.connecting_wires_list[outlet_index]].LPN_solution_ids[1]], # outlet flow
                areas[inlet_index], #inlet area
                areas[outlet_index], # outlet area
                self.angles[outlet_index], # direction change
                self.junction_length[outlet_index], # junction length
                self.num_outlets, # number of outlets
                self.flow_time_der[outlet_index]]) # flow time derivative
        self.mat["C"]  =  [-1333.2*z_scaler_out.inverse_transform(DNN_model.predict(
            z_scaler_in.fit_transform(np.reshape(
            DNN_input[outlet_index], (1,11))))) for outlet_index in outlet_indices] + [0]

        # SET dC MATRIX
        DNN_derivs_all = [2*len(self.flow_directions)*(0,)]*len(self.mat["C"])
        for outlet_index in outlet_indices:
            DNN_derivs = 2*len(self.flow_directions)*[0,]
            xt = tf.convert_to_tensor(DNN_input[outlet_index].reshape((1,DNN_input[outlet_index].size)))
            with tf.GradientTape() as g:
                g.watch(xt)
                y = DNN_model(xt)
            grads = g.jacobian(y, xt).numpy().squeeze()
            dP_in = grads[0]
            dQ_in = grads[3]
            dQ_out = grads[4]
            DNN_derivs[2*inlet_index] = - dP_in
            DNN_derivs[2*inlet_index+1] = -1333.2*dQ_in
            DNN_derivs[2*outlet_index+1] = -1333.2*dQ_out
            DNN_derivs_all[outlet_indices.index(outlet_index)]= tuple(DNN_derivs)
        self.mat["dC"] = DNN_derivs_all

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