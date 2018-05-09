# -*- coding: utf-8 -*-
"""
Created on Wed May  9 12:19:16 2018

@author: Bob
"""
import Nets
import numpy as np
import torch
from torch import nn


def Single_test(Net, model_params, train_input, train_target, test_input, test_target, \
                nb_iters = 100, batch_size = 316, use_tol = False, inter_states = False):
    """ 
        This function estimates the errors of the test and train data on the netowrk 'Net', trained by the train data.
    Arguments: 
        Net: the neural network structure to be used, type nn.Module
        model_params: the models / networks parameters, such as size, chn_conv, ker_conv, ker_pool, nb_hidden, len_IN_lin.
                    see the networks specification for more detail
        train_input: the input to train the network, type Variable() of size N*C*L
                    where N is the number of samples, C is the number of channels and L is the length of a sample
        train_target: the target to train the network
        test_input: the input to be tested
        test_target: the target of the test data (not used for training)
        nb_iters: (max) number of iterations during training
        batch_size: size of the batches during training
        use_tol = if True, use the tolerance on the loss as stopping criterion
        inter_states = if True, compute the train and test errors every 25 iterations
    Output:
        train_error: The error of the training data at the last iteration
        test_error: The error of the testing data at the last iteration
        err_array: If inter_states is True, returns an array containing a row with information at intervals of 25 iteration:
            first column: the iteration, second column: the train error, third column: the test error
    """
    
    #define the model and the criterion
    model = Net(*model_params)
    criterion = nn.CrossEntropyLoss()

    #Use cuda if available
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
        
    #standardize the data
    mu, std = train_input.data.mean(), train_input.data.std()
    train_input.data.sub_(mu).div_(std)
    test_input.data.sub_(mu).div_(std)

    #apply the model (train the network)
    err_array = Nets.train_model(model, criterion, train_input, train_target, \
                                 nb_iters, batch_size, use_tol, inter_states, test_input, test_target)
    #compute the errors
    train_error = Nets.compute_nb_errors(model, train_input, train_target, batch_size) / train_input.size(0) * 100
    test_error = Nets.compute_nb_errors(model, test_input, test_target, batch_size) / test_input.size(0) * 100
    print('train_error {:.02f}; test_error {:.02f}'.format(train_error,test_error))
    return train_error, test_error, err_array


def Cross_validation_mxk(Net, model_params, data_input, data_target, \
                         m_fold = 1, k_fold = 1, nb_iters = 100, batch_size = 316, use_tol = False, inter_states = False):
    
    """ 
        This functions runs an mxk (also called mxn) cross-validation on the given data. It returns the estimated errors.
    Arguments: 
        Net: the neural network structure to be used, type nn.Module
        model_params: the models / networks parameters, such as size, chn_conv, ker_conv, ker_pool, nb_hidden, len_IN_lin.
                    see the networks specification for more detail
        data_input: the input that will be split for cross validation, type Variable() of size N*C*L
                    where N is the number of samples, C is the number of channels and L is the length of a sample
        data_target: the target for the cross-validation the network
        m_fold: the number of times the whole k-fold-crossvalidation is performed 
        k_fold: the number of folds in a cross-validation
        test_input: the input to be tested
        test_target: the target of the test data (not used for training)
        nb_iters: (max) number of iterations during training
        batch_size: size of the batches during training
        use_tol = if True, use the tolerance on the loss as stopping criterion
        inter_states = if True, compute the train and test errors every 25 iterations
    Output:
        train_error: The error of the training data at the last iteration
        test_error: The error of the testing data at the last iteration
        err_array: If inter_states is True, returns an array containing a row with information at intervals of 25 iteration:
            first column: the iteration, second column: the train error, third column: the test error
    """
    
    #define the criterion
    criterion = nn.CrossEntropyLoss().cuda()
    
    #define the train and validation set errors
    err_array = [[0,]*k_fold]*m_fold
    trn_error = np.zeros((m_fold, k_fold))
    vldt_error = np.zeros((m_fold, k_fold))
    
    
    num_row = data_input.shape[0]
    interval = int(num_row / k_fold)

    for m in range(0,m_fold):
    #prepare the indexing arrays
        np.random.seed(m)
        indices = np.random.permutation(num_row)
        k_indices = np.array([indices[k * interval: (k + 1) * interval]
                     for k in range(k_fold)])
        k_list=np.arange(k_indices.shape[0])  

        for k in range(0,k_fold):
            #define the model, done for each k_fold in order to reset it and its parameters
            model = Net(*model_params)    
            #Use cuda if available
            if torch.cuda.is_available():
                model.cuda()
                criterion.cuda()
                
            trn_target = data_target[np.ravel(k_indices[k_list[k*np.ones(len(k_list))!=k_list]])]
            trn_input = data_input[np.ravel(k_indices[k_list[k*np.ones(len(k_list))!=k_list]])]
            vldt_target = data_target[k_indices[k]]
            vldt_input = data_input[k_indices[k]]
    
            #standardize the data
            mu, std = trn_input.data.mean(), trn_input.data.std()
            trn_input.data.sub_(mu).div_(std)
            vldt_input.data.sub_(mu).div_(std)
            
            #apply the model
            err_array[m][k] = Nets.train_model(model, criterion, trn_input, trn_target, \
                     nb_iters, batch_size, use_tol, inter_states, vldt_input, vldt_target)
            #compute the errors
            trn_error[m,k] = compute_nb_errors(model, trn_input, trn_target, batch_size) / trn_input.size(0) * 100
            vldt_error[m,k] = compute_nb_errors(model, vldt_input, vldt_target, batch_size) / vldt_input.size(0) * 100
            print('Fold {:0d}: train_error {:.02f}; validation_error {:.02f}'.format(k, trn_error[m,k],vldt_error[m,k]))
    return trn_error, vldt_error, err_array

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        if b+mini_batch_size > data_input.size(0):
            mini_batch_size_var = data_input.size(0)-b
        else: mini_batch_size_var = mini_batch_size
        model.eval()
        output = model(data_input.narrow(0, b, mini_batch_size_var))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(0, mini_batch_size_var):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors