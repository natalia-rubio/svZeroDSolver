#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
August 2021
@author: Natalia Rubio
Implementation of Mynard (2015) Junction Loss Model
(see accompanying pdf for details)

Inputs: U, A, theta --> (m+n)x1 numpy arrays containing the flow velocity,
area, and angle of the junction branches.

Outputs: C, K --> numpy arrays of the C and K loss coefficients
"""
import autograd.numpy as np
import pdb
def wrap_to_pi(angle):
    # function to map angles to a magnitude less than pi
    #pdb.set_trace()
    while abs(angle) > np.pi:
        if angle > 0:
            angle = angle - 2*np.pi
        else:
            angle = angle + 2*np.pi
    return angle
wrap_to_pi = np.vectorize(wrap_to_pi)

def wrap_to_2pi(angle):
    # function to map angles to a magnitude less than 2pi
    while abs(angle) > 2*np.pi:
        if angle > 0:
            angle = angle - 4*np.pi
        else:
            angle = angle + 4*np.pi
    return angle
wrap_to_2pi = np.vectorize(wrap_to_2pi)

def junction_loss_coeff(U, A, theta):
    theta = wrap_to_pi(theta)
    flow_rate = np.multiply(U, A) # flow rate
    inlets = (flow_rate >= 0).astype(bool) # identify inlets
    outlets = (flow_rate < 0).astype(bool) # identify outlets
    # Angle Manipulations ------------------------------------------------

    pseudo_outlet_angle = np.mean(theta[outlets]) # initialize pseudo-outlet angle

    pseudo_inlet_angle = np.arctan2(
            np.sum(np.multiply(flow_rate[inlets], np.sin(theta[inlets]))),
            np.sum(np.multiply(flow_rate[inlets], np.cos(theta[inlets])))
            )# initialize pseudo-inlet angle
    if abs(pseudo_inlet_angle - pseudo_outlet_angle) < np.pi/2:
        pseudo_outlet_angle += np.pi # enforce that the pseudo-outlet angle be in the second quadrant

    theta = wrap_to_pi(theta - pseudo_outlet_angle) # set the average outlet angle to zero
    if np.mean(np.multiply(
            flow_rate[inlets], np.sin(theta[inlets]))) < 0:
        theta = -theta # enforce that the majority of flow comes from positive angles
    pseudo_inlet_angle = np.arctan2(
            np.sum(np.multiply(flow_rate[inlets], np.sin(np.abs(theta[inlets])))),
            np.sum(np.multiply(flow_rate[inlets], np.cos(np.abs(theta[inlets]))))
                    ) # initialize pseudo-inlet angle

    # Calculated Junction Parameters ------------------------------------

    flow_rate_total = np.sum(flow_rate[inlets]) # total flow rate
    flow_ratio = np.divide(
            -1*flow_rate[outlets], flow_rate_total) # flow ratio (lambda)
    energy_transfer_factor = np.multiply(
            np.multiply(0.8*np.subtract(np.pi, pseudo_inlet_angle),
                        np.sign(theta[outlets])) - 0.2,
            np.subtract(1,flow_ratio)) # energy transfer factor (eta)

    total_pseudo_area = np.divide(flow_rate_total, np.multiply(
            np.subtract(1,energy_transfer_factor),
            np.sum(np.multiply(U[inlets], flow_rate[inlets]))/flow_rate_total)) # total_pseudo_area (A')
    area_ratio = np.divide(total_pseudo_area, A[outlets]) # area ratio (psi)
    phi = wrap_to_2pi(np.subtract(pseudo_inlet_angle, theta[outlets])) # angle deviation (phi)
    # Calculate Loss Coefficients ---------------------------------------

    C = np.zeros((np.size(U),))
    print("flow ratio: ", flow_ratio)
    C[outlets] = np.multiply(
            (1-np.exp(-flow_ratio/0.02)),
            np.subtract(1, np.divide(np.cos(0.75*(np.subtract(np.pi, phi))),
                                     np.multiply(area_ratio, flow_ratio)))) # compute C
    K = np.NaN
    if np.size(U) <= 3: # check that there are three or fewer branches
        if np.sum(outlets) == 1: # check for converging flow
            U_common = U[outlets]
            K = np.multiply(np.divide(U[outlets]**2, U_common**2),
                        (2*C[outlets] + np.divide(U[inlets]**2,
                         U[outlets]**2) - 1)) # compute K
        elif np.sum(inlets) == 1: # check for diverging flow
            U_common = U[inlets]
            K = np.multiply(np.divide(U[outlets]**2, U_common**2),
                        (2*C[outlets] + np.divide(U[inlets]**2,
                         U[outlets]**2) - 1)) # compute K
        # pdb.set_trace()


    return (C, K, energy_transfer_factor)
