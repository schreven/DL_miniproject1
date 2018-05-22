# -*- coding: utf-8 -*-
"""
Created on Tue May  8 18:10:08 2018

@author: Bob
"""

import matplotlib.pyplot as plt
import numpy as np
import pylab
import os
import pylab

def plot_interstates(err_array):    
    
    err_mean = np.mean(err_array, axis= (0,1))
    err_std = np.std(err_array, axis = (0,1))
    fig, ax = plt.subplots()
    ax.errorbar(np.arange(0,25*len(err_mean),25),err_mean[:,1], yerr = err_std[:,1])
    ax.errorbar(np.arange(0,25*len(err_mean),25),err_mean[:,2], yerr = err_std[:,2])
    ax.set_xlabel('iteration [u]')
    ax.set_ylabel('error [%]')
    ax.set_title('Errors: mean and std')
    ax.legend(['train error','test error'])
    fig.show()
    pylab.savefig(os.path.join('arrays_and_images','solo_fourth.png'))
    
    
def plot_first():
    err_list = np.load(os.path.join('arrays_and_images','first_err_list.npy'))
    title_list = ['batch_size: 1','batch_size: 5','batch_size: 50','batch_size: 316']
    fig, axs = plt.subplots(2,2,figsize=(10,5), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.2)
    for i_ in range(len(err_list)):
        err_mean = np.mean(err_list[i_], axis= (0,1))
        err_std = np.std(err_list[i_], axis = (0,1))
        axs[i_//2][i_%2].errorbar(np.arange(0,25*len(err_mean),25),err_mean[:,1], yerr = err_std[:,1])
        axs[i_//2][i_%2].errorbar(np.arange(0,25*len(err_mean),25),err_mean[:,2], yerr = err_std[:,2])
        if i_%2 == 0:
            axs[i_//2][i_%2].set_ylabel('error [%]')   
        if i_//2 == 1:
            axs[i_//2][i_%2].set_xlabel('iteration [u]')
        axs[i_//2][i_%2].set_title(title_list[i_])
        axs[i_//2][i_%2].legend(['train error','test error'])
    fig.show()

    pylab.savefig(os.path.join('arrays_and_images','first_vis.png'))
    

    
    
def plot_second():
    err_list = np.load(os.path.join('arrays_and_images','second_err_list.npy'))
    title_list = ['nb_hidden: 5','nb_hidden: 20','nb_hidden: 100','nb_hidden: 500']
    fig, axs = plt.subplots(2,2,figsize=(10,5), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.2)
    for i_ in range(len(err_list)):
        err_mean = np.mean(err_list[i_], axis= (0,1))
        err_std = np.std(err_list[i_], axis = (0,1))
        axs[i_//2][i_%2].errorbar(np.arange(0,25*len(err_mean),25),err_mean[:,1], yerr = err_std[:,1])
        axs[i_//2][i_%2].errorbar(np.arange(0,25*len(err_mean),25),err_mean[:,2], yerr = err_std[:,2])
        if i_%2 == 0:
            axs[i_//2][i_%2].set_ylabel('error [%]')   
        if i_//2 == 1:
            axs[i_//2][i_%2].set_xlabel('iteration [u]')
        axs[i_//2][i_%2].set_title(title_list[i_])
        axs[i_//2][i_%2].legend(['train error','test error'])
    fig.show()

    pylab.savefig(os.path.join('arrays_and_images','second_vis.png'))
    

