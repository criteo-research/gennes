#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 10:29:46 2018

@author: l.faury
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def mesh_vizu(f, xlim, ylim, scope=0):
    """ Plot a contour plot of function f
    Args:
        function: functions to plot contours for
        xlim : [-x,x]
        ylim : [-y,y] limits the plot
        scope = string, name
    Returns:
        ax, CS
    """
    fig = plt.figure(scope)
    ax = fig.add_subplot(1, 1, 1)
    x, y = np.mgrid[xlim[0]:xlim[1]:0.1, ylim[0]:ylim[1]:0.1]
    CS = []
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = f(np.array([[x[i, j], y[i, j]]]))
    cs = ax.contour(x, y, z, levels=np.arange(np.min(z), np.max(z), 10.0), cmap='autumn')
    ax.contourf(x, y, z, cmap='autumn', levels=np.arange(np.min(z), np.max(z), 10.0), alpha=0.2, linestyles='')
    CS.append(cs)
    return ax, CS


def plot_perf(mins, pop_size):
    """ Plots the simple regret performances
    Args:
        mins: np array
        pop_size: int, population size
    """
    size = np.size(mins)
    plt.figure('fvalues')
    ax = plt.subplot(111)
    ax.set_xscale('log', nonposx='clip')
    ax.set_yscale('log', nonposy='clip')
    ax.set_xlabel('function evaluation')
    ax.set_ylabel('function value')
    ax.plot(pop_size*np.arange(1, size+1), np.minimum.accumulate(mins),
            color='green', label='gennes')
    ax.legend()
    plt.savefig(os.path.join('img', 'fvalues'))


def plot_contours(session, bbo, popi):
    """ Plots the contours and populations along the optimization procedure, on the
    compact [-3,3]^2
    Args:
        session: TensorFlow session
        bbo: ObjFunction instance, objective
        popi: array, populations individuals
    """
    def f(x):
        return bbo.bbox_oracle0(session, x)

    ax, CS = mesh_vizu(f, [-3, 3], [-3, 3], 'fplot')
    ax.scatter(bbo.min[0], bbo.min[1], marker='o', c='yellow',
               label='global minimum', edgecolor='black')
    for i, p in enumerate(popi):
        scatter = ax.scatter(p[:, 0], p[:, 1], marker='+',
                             label='query', c='blue', s=20, zorder=1)
        ax.legend(loc=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(os.path.join('img', 'fplot'+str(i)))
        scatter.remove()
