# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 16:02:13 2018

@author: Bob
"""
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch import FloatTensor as Tensor

import matplotlib.pyplot as plt
import numpy as np

#from sweeps.py import Sweep_nb_hidden_batch_size 
import Nets
import Validate
import Visualizations as Vis

import dlc_bci as bci
train_input , train_target = bci.load(root = './ data_bci')
print ('train_input:', str ( type ( train_input ) ) , train_input.size () )
print ('train_target:', str ( type ( train_target ) ) , train_target.size () )
test_input , test_target = bci.load ( root = './data˙bci' , train = False )
print ('test_input:', str ( type ( test_input ) ) , test_input.size () )
print ('test target:', str ( type ( test_target ) ) , test_target.size () )

# Setting the data to the Variable type
train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

# Use cuda if available
if torch.cuda.is_available():
    train_input, train_target = train_input.cuda(), train_target.cuda()
    test_input, test_target = test_input.cuda(), test_target.cuda()

size = train_input.size() 
    

 



#### PARAMETERS AND SHIT

###NET 1
Net = Nets.Net1

nb_hidden = 1000
model_params = [size, nb_hidden]


m_fold = 2                  #number of folds of full k_fold cross validations
k_fold = 5                  #number of folds of the cross-validation
nb_iters = 1000             #(max) number of iterations during training
batch_size = 316            #size of the batches during training
use_tol = False             #use the tolerance as stopping criterion or not         TODOOO: add tolerance argument or remove this one
inter_states = True         #compute the train and test errors every 25 iterations, print them and save them in err_array

crossval_params = [m_fold, k_fold, nb_iters, batch_size, use_tol, inter_states]
singleval_params = [nb_iters, batch_size, use_tol, inter_states]

#Apply crossvalidation
trn_error, vldt_error, err_array = Validate.Cross_validation_mxk(Net, model_params, train_input, train_target, *crossval_params)

mean_vldt = np.mean(vldt_error)
std_vldt = np.std(vldt_error)

err_array_mean = np.mean(err_array, axis = (0,1))
err_array_std = np.std(err_array, axis = (0,1))

Vis.plot_interstates(err_array_mean, err_array_std)



###NET 2
Net = Nets.Net1

chn_conv = [50,200]          #convolutional layers, nb of channels at output
ker_conv = [5,4]             #convolutional layers, kernel sizes
ker_pool = [2,2]              #max_pool filter, kernel sizes
nb_hidden = [200]          # nb hidden neurons in linear layers
len_IN_lin = chn_conv[-1]*(((size[2]-ker_conv[0]+1)/ker_pool[0])-ker_conv[1]+1)/ker_pool[1]

model_params = [size, chn_conv, ker_conv, ker_pool, nb_hidden, len_IN_lin]

m_fold = 2                  #number of folds of full k_fold cross validations
k_fold = 5                  #number of folds of the cross-validation
nb_iters = 1000             #(max) number of iterations during training
batch_size = 316       #size of the batches during training
use_tol = False             #use the tolerance as stopping criterion or not         TODOOO: add tolerance argument or remove this one
inter_states = True         #compute the train and test errors every 25 iterations, print them and save them in err_array

crossval_params = [m_fold, k_fold, nb_iters, batch_size, use_tol, inter_states]
singleval_params = [nb_iters, batch_size, use_tol, inter_states]

#Apply training and testing once on given data
#trn_error, vldt_error, err_array = Validate.Single_test(Nets.Net2, model_params, train_input, train_target,
#                                                        test_input, test_target, *singleval_params)
#Apply crossvalidation
#trn_error, vldt_error, err_array = Validate.Cross_validation_mxk(Nets.Net2, model_params, train_input, train_target, *crossval_params)



#mean_vldt = np.mean(vldt_error)
#std_vldt = np.std(vldt_error)
#
#err_array_mean = np.mean(err_array, axis = (0,1))
#err_array_std = np.std(err_array, axis = (0,1))
#
#Vis.plot_interstates(err_array_mean, err_array_std)


### NET 3
#chn_conv = [50,100,200]          #convolutional layers, nb of channels at output
#ker_conv = [5,4,3]             #convolutional layers, kernel sizes
#ker_pool = [2,2,2]              #max_pool filter, kernel sizes
#nb_hidden = [1000,250,50,10]    # nb hidden neurons in linear layers
#len_IN_lin = chn_conv[-1]*((size[2]-ker_conv[0]+1)/ker_pool[0])

#print(mean_vldt, std_vldt)
##SWEEEEEEPSs
#nb_hiddens = [[50], [200], [500], [1000]]
#mini_batch_sizes = [40,316]
##
#def Sweep_nb_hidden_batch_size(nb_hiddens, mini_batch_sizes):
#    train_error = np.zeros([len(nb_hiddens),len(mini_batch_sizes)])
#    test_error = np.zeros([len(nb_hiddens),len(mini_batch_sizes)])
#    
#    for h_, nb_hidden in enumerate(nb_hiddens):
#        model = Net2(size, nb_hidden, chn_conv, ker_conv, ker_pool, len_IN_lin).cuda()
#        for s_, mini_batch_size in enumerate(mini_batch_sizes):
#            train_model(model, criterion, train_input, train_target, mini_batch_size, False)
#            train_error[h_,s_] = compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100
#            test_error[h_,s_] = compute_nb_errors(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100
#            print('train_error {:.02f}; test_error {:.02f}'.format(train_error[h_,s_],test_error[h_,s_]))
#        
#    for s_ in range(len(mini_batch_sizes)):
#        plt.plot(nb_hiddens,train_error[:,s_])
#        plt.plot(nb_hiddens,test_error[:,s_])
#    legends = np.append(['tr, size:{0:0d}'.format(si) for si in mini_batch_sizes],['te, size:{0:0d}'.format(si) for si in mini_batch_sizes])
#    plt.legend(legends)
#
#Sweep_nb_hidden_batch_size(nb_hiddens,mini_batch_sizes)
    
    
#if len(ker_conv) == 1:
#    len_IN_lin = chn_conv[-1]*((size[2]-ker_conv[0]+1)/ker_pool[0])
#    if len_IN_lin/chn_conv[-1] % 1 !=0:
#        print("Given layer sizes are not valid")
#
#if len(ker_conv) == 2:
#    len_IN_lin = chn_conv[-1]*(((size[2]-ker_conv[0]+1)/ker_pool[0])-ker_conv[1]+1)/ker_pool[1]
#    if len_IN_lin/chn_conv[-1] % 1 !=0:
#        print("Given layer sizes are not valid")
#if len(ker_conv) == 3:
#    len_IN_lin = chn_conv[-1]*(((((size[2]-ker_conv[0]+1)/ker_pool[0])-ker_conv[1]+1)/ker_pool[1])-ker_conv[2]+1)/ker_pool[2]
#    if len_IN_lin/chn_conv[-1] % 1 !=0:
#        print("Given layer sizes are not valid")
#