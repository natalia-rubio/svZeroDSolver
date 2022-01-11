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

import tensorflow as tf
import tensorflow.math as tfm


pi = tf.constant(3.14159, dtype = tf.float64)
def wrap_to_pi(angle):
    # function to map angles to a magnitude less than pi
    cond = lambda angle: tfm.greater(tfm.abs(angle), pi)
    fn_true = lambda: angle - 2*pi
    fn_false = lambda: angle + 2*pi
    body = lambda angle: tf.cond(tfm.greater(angle, pi), fn_true, fn_false)
    tf.while_loop(cond, body, [angle])
    return angle

def wrap_to_2pi(angle):
    # function to map angles to a magnitude less than pi
    cond = lambda angle: tfm.greater(tfm.abs(angle), pi)
    fn_true = lambda: angle - 4*pi
    fn_false = lambda: angle + 4*pi
    body = lambda angle: tf.cond(tfm.greater(angle, pi), fn_true, fn_false)
    tf.while_loop(cond, body, [angle])
    return angle

def junction_loss_coeff_tf(U, A, theta):
    theta = tf.map_fn(wrap_to_pi, theta)
    flow_rate = tfm.multiply(U, A) # flow rate
    inlets = tf.greater_equal(flow_rate, tf.constant([0], dtype= "float64")) # identify inlets
    outlets = tf.less(flow_rate, tf.constant([0], dtype= "float64")) # identify outlets

    # Angle Manipulations ------------------------------------------------

    pseudo_outlet_angle = tfm.reduce_mean(tf.boolean_mask(theta,outlets)) # initialize pseudo-outlet angle
    pseudo_inlet_angle = tfm.atan2(
            tfm.reduce_sum(tfm.multiply(tf.boolean_mask(flow_rate, inlets),
            tfm.sin(tf.boolean_mask(theta,inlets)))),
            tfm.reduce_sum(tfm.multiply(tf.boolean_mask(flow_rate, inlets),
            tfm.cos(tf.boolean_mask(theta,inlets)))))# initialize pseudo-inlet angle

    tf.cond(tfm.less(tfm.abs(pseudo_inlet_angle - pseudo_outlet_angle), pi/2),
            lambda: pseudo_outlet_angle + pi, lambda: pseudo_outlet_angle) # enforce that the pseudo-outlet angle be in the second quadrant

    theta = tf.map_fn(wrap_to_pi, theta - pseudo_outlet_angle) # set the average outlet angle to zero

    tf.cond(tfm.less(tfm.reduce_mean(tfm.multiply(tf.boolean_mask(
        flow_rate,inlets), tfm.sin(tf.boolean_mask(theta, inlets)))), 0),
        lambda: -theta, lambda: theta) # enforce that the majority of flow comes from positive angles

    pseudo_inlet_angle = tfm.atan2(
            tfm.reduce_sum(tfm.multiply(tf.boolean_mask(flow_rate, inlets),
            tfm.sin(tfm.abs(tf.boolean_mask(theta,inlets))))),
            tfm.reduce_sum(tfm.multiply(tf.boolean_mask(flow_rate, inlets),
            tfm.cos(tfm.abs(tf.boolean_mask(theta, inlets)))))) # initialize pseudo-inlet angle

    # Calculated Junction Parameters ------------------------------------

    flow_rate_total = tfm.reduce_sum(tf.boolean_mask(flow_rate, inlets)) # total flow rate
    flow_ratio = tfm.divide(
            -1*tf.boolean_mask(flow_rate,outlets), flow_rate_total) # flow ratio (lambda)

    energy_transfer_factor = tfm.multiply(
            tfm.multiply(0.8*tfm.subtract(pi, pseudo_inlet_angle),
            tfm.sign(tf.boolean_mask(theta, outlets))) - 0.2,
            (1-flow_ratio)) # energy transfer factor (eta)

    total_pseudo_area = tfm.divide(flow_rate_total, tfm.multiply(
            (1-energy_transfer_factor),
            tfm.reduce_sum(tfm.multiply(tf.boolean_mask(U,inlets),
            tf.boolean_mask(flow_rate,inlets)))/flow_rate_total)) # total_pseudo_area (A')
    area_ratio = tfm.divide(total_pseudo_area, tf.boolean_mask(A,outlets)) # area ratio (psi)
    phi = tf.map_fn(wrap_to_2pi, tfm.subtract(pseudo_inlet_angle, theta[outlets])) # angle deviation (phi)

    # Calculate Loss Coefficients ---------------------------------------

    C_outlets = tfm.multiply((1-tfm.exp(-flow_ratio/0.02)),
            (1-tfm.divide(tfm.cos(0.75*(pi - phi)),
            tfm.multiply(area_ratio, flow_ratio)))) # compute C

    return (C_outlets, outlets)







