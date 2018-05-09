# -*- coding: utf-8 -*-
"""
Created on Tue May  8 18:10:08 2018

@author: Bob
"""

import matplotlib.pyplot as plt
import numpy as np

err_array_1 = np.load('err_array_1.npy')
err_array_2 = np.load('err_array_2.npy')
err_array_3 = np.load('err_array_3.npy')

def plot_sweep(arr1, arr2, arr3):
    plt.plot(err_array_1[:,0],err_array_1[:,1])
    plt.plot(err_array_1[:,0],err_array_1[:,2])
    plt.plot(err_array_2[:,0],err_array_2[:,1])
    plt.plot(err_array_2[:,0],err_array_2[:,2])
    plt.plot(err_array_3[:,0],err_array_3[:,1])
    plt.plot(err_array_3[:,0],err_array_3[:,2])
    plt.legend(('train, changing eta','test, changing eta', 'train, eta = 1e-3', 'test, eta = 1e-1',\
                'train, eta= 0.2/iter', 'test, eta=0.2/iter'))

def plot_interstates(err_mean, err_std):
    fig, ax = plt.subplots()
    ax.errorbar(err_mean[:,0],err_mean[:,1], yerr = err_std[:,1])
    ax.errorbar(err_mean[:,0],err_mean[:,2], yerr = err_std[:,2])
    ax.set_xlabel('iteration [u]')
    ax.set_ylabel('error [%]')
    ax.set_title('Errors: mean and std')
    ax.legend(['train error','test error'])
    fig.show()