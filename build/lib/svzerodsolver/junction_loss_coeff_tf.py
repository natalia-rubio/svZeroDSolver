#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
August 2021
@author: Natalia Rubio
Implementation of Mynard (2015) Junction Loss Model
(see accompanying pdf for details)

Itfmuts: U, A, theta --> (m+n)x1 numpy arrays containing the flow velocity,
area, and angle of the junction branches.

Outputs: C, K --> numpy arrays of the C and K loss coefficients
"""
import numpy as np
import tensorflow as tf
import tensorflow.math as tfm
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

def junction_loss_coeff_tf(U, A, theta):
    #pdb.set_trace()
    theta = wrap_to_pi(theta)
    flow_rate = tfm.multiply(U, A) # flow rate
    flow_rate_np = np.multiply(U, A)
    inlets = (flow_rate_np >= 0).astype(bool) # identify inlets
    outlets = (flow_rate_np < 0).astype(bool) # identify outlets
    #pdb.set_trace()
    # Angle Manipulations ------------------------------------------------

    pseudo_outlet_angle = np.mean(theta[outlets]) # initialize pseudo-outlet angle

    pseudo_inlet_angle = tfm.atan2(
            tfm.reduce_sum(tfm.multiply(flow_rate[inlets], tfm.sin(theta[inlets]))),
            tfm.reduce_sum(tfm.multiply(flow_rate[inlets], tfm.cos(theta[inlets])))
            )# initialize pseudo-inlet angle
    if abs(pseudo_inlet_angle - pseudo_outlet_angle) < np.pi/2:
        pseudo_outlet_angle += np.pi # enforce that the pseudo-outlet angle be in the second quadrant

    theta = wrap_to_pi(theta - pseudo_outlet_angle) # set the average outlet angle to zero
    if tfm.reduce_mean(tfm.multiply(
            flow_rate[inlets], tfm.sin(theta[inlets]))) < 0:
        theta = -theta # enforce that the majority of flow comes from positive angles
    pseudo_inlet_angle = tfm.atan2(
            tfm.reduce_sum(tfm.multiply(flow_rate[inlets], tfm.sin(tfm.abs(theta[inlets])))),
            tfm.reduce_sum(tfm.multiply(flow_rate[inlets], tfm.cos(tfm.abs(theta[inlets]))))
                    ) # initialize pseudo-inlet angle

    # Calculated Junction Parameters ------------------------------------

    flow_rate_total = tfm.reduce_sum(flow_rate[inlets]) # total flow rate
    flow_ratio = tfm.divide(
            -1*flow_rate[outlets], flow_rate_total) # flow ratio (lambda)
    energy_transfer_factor = tfm.multiply(
            tfm.multiply(0.8*tfm.subtract(np.pi, pseudo_inlet_angle),
                        tfm.sign(theta[outlets])) - 0.2,
            tfm.subtract(1,flow_ratio)) # energy transfer factor (eta)

    total_pseudo_area = tfm.divide(flow_rate_total, tfm.multiply(
            tfm.subtract(1,energy_transfer_factor),
            tfm.reduce_sum(tfm.multiply(U[inlets], flow_rate[inlets]))/flow_rate_total)) # total_pseudo_area (A')
    area_ratio = tfm.divide(total_pseudo_area, A[outlets]) # area ratio (psi)
    phi = wrap_to_2pi(tfm.subtract(pseudo_inlet_angle, theta[outlets])) # angle deviation (phi)

    # Calculate Loss Coefficients ---------------------------------------

    C_outlets = tfm.multiply(
            (1-tfm.exp(-flow_ratio/0.02)),
            tfm.subtract(1, tfm.divide(tfm.cos(0.75*(tf.cast(tfm.subtract(np.pi, phi),dtype=tf.float64))),
                                       tfm.multiply(area_ratio, flow_ratio)))) # compute C

    return (C_outlets, outlets)







