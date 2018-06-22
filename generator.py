#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 23:50:07 2018

@author: l.faury
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


# Hard coded value for code simplicity -- should be adaptated if the problem changes drastically
DOMAIN = [(-3, 3)] # compact set of interest
DEPTH = 5 # generator depth
OUTVAR = 1.0 # variance of the ouput weights
WIDTH = 128 # generator width

class Generator(object):
    """ Generative Neural Network model for global optimization use within GENNES. """

    def __init__(self, noise_dim, bb_dim, step_size, pop_size):
        """ Init function, builds the graph

        Args:
            noise_dim : int, dimensionality of the input noise
            bb_dim : int, dimensionality of the black-box problem (e.g dim of the output)
            step_size : float, learning rate for the generator
            pop_size : int, population size
        """
        self.bb_dim = bb_dim
        self.noise_dim = noise_dim
        self.step_size = step_size
        self.pop_size = pop_size
        self.widths = np.zeros((self.bb_dim,))
        self.shifts = np.zeros((self.bb_dim,))
        domain = DOMAIN*self.bb_dim
        for i, d in enumerate(domain):
            self.widths[i] = 0.5*(d[1]-d[0])
            self.shifts[i] = 0.5*(d[1]+d[0])
        self._create_generator()

    def _create_generator(self):
        """ Builds the computational graph"""
        # Creating placeholders and constants
        with tf.name_scope('placeholders'):
            self.x = tf.placeholder(tf.float64, [None, self.noise_dim], name='in')
            self.fgrad = tf.placeholder(tf.float64, [None, self.bb_dim],
                                        name='fgrad')  # optimized functions gradients
            self.alpha = tf.constant(0.2, dtype=tf.float64, name='alpha')  # coefficient for leaky relus

        # Creating model
        with tf.name_scope('layers'):
            net = self.x
            with slim.arg_scope([slim.fully_connected], activation_fn=None):
                for _ in range(DEPTH):
                    net = slim.fully_connected(net, WIDTH)
                    net = tf.nn.leaky_relu(net, self.alpha)

            # adding the final layer
            weights_initializer = tf.truncated_normal_initializer(stddev=OUTVAR)
            net = slim.fully_connected(net, self.bb_dim, activation_fn=tf.nn.tanh,
                                       weights_initializer=weights_initializer, scope='fcout')

            # imposing the support to lay in domain
            net = self.shifts + tf.multiply(self.widths, net)

        # Creating output
        with tf.name_scope('output'):
            self.y = tf.identity(net, name='out')

        # Creating the gradient and update operation
        with tf.name_scope('ops'):
            self.params = tf.trainable_variables()
            self.unormgradient = tf.gradients(
                self.y, self.params, self.fgrad)  # generator gradient
            self.normgradient = list(
                map(lambda x: tf.div(x, self.pop_size), self.unormgradient))  # generator gradient (normalized)

            self.optimizer = tf.train.AdamOptimizer(self.step_size)
            self.update = self.optimizer.apply_gradients(
                zip(self.normgradient, self.params))
