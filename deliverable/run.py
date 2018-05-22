# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 16:02:13 2018

@author: Bob
"""
import torch
from torch.autograd import Variable
from torch import nn

import Nets
import Validate
import Visualizations as Vis
import dlc_bci as bci


###Fetch the data
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
    



###NET 1


Net = Nets.Net1             #Net1 as defined in Nets.py has just on linear hidden layer
nb_hidden = 5
model_params = [size, nb_hidden]


m_fold = 1                  #number of folds of full k_fold cross validations
k_fold = 5                 #number of folds of the cross-validation
nb_iters = 1000             #(max) number of iterations during training
batch_size = 5          #size of the batches during training
use_tol = True             #use the tolerance as stopping criterion or not         
inter_states = True         #compute the train and test errors every 25 iterations, print them and save them in err_array

crossval_params = [m_fold, k_fold, nb_iters, batch_size, use_tol, inter_states]
singleval_params = [nb_iters, batch_size, use_tol, inter_states]

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD#


#Apply training and testing once on given data
trn_error, vldt_error, err_array = Validate.Single_test(Nets.Net2, model_params, train_input, train_target,
                                                        test_input, test_target, *singleval_params)
#Apply crossvalidation
trn_error, vldt_error, err_array = Validate.Cross_validation_mxk(Net, model_params, criterion, optimizer, train_input, train_target, *crossval_params)

Vis.plot_interstates(err_array)



    

###NET 2
Net = Nets.Net2             #Nets2 has following operations: one layers of : conv -> max_pool -> Relu -> drop_out
                                            #two layer of : Lin -> ReLu-> drop_out
                                            # one final Lin

chn_conv = [50]          #convolutional layers, nb of channels at output
ker_conv = [5]             #convolutional layers, kernel sizes
ker_pool = [2]              #max_pool filter, kernel sizes
nb_hidden = [15,5]          # nb hidden neurons in linear layers

len_IN_lin = chn_conv[-1]*((size[2]-ker_conv[0]+1)/ker_pool[0])


model_params = [size, chn_conv, ker_conv, ker_pool, nb_hidden, len_IN_lin]

m_fold = 2                  #number of folds of full k_fold cross validations
k_fold = 5                  #number of folds of the cross-validation
nb_iters = 400             #(max) number of iterations during training
batch_size = 5      #size of the batches during training
use_tol = False            #use the tolerance as stopping criterion or not         
inter_states = True         #compute the train and test errors every 25 iterations, print them and save them in err_array

crossval_params = [m_fold, k_fold, nb_iters, batch_size, use_tol, inter_states]
singleval_params = [nb_iters, batch_size, use_tol, inter_states]

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam

#Apply training and testing once on given data
#trn_error, vldt_error, err_array = Validate.Single_test(Nets.Net2, model_params, train_input, train_target, test_input, test_target, *singleval_params)
#Apply crossvalidation
trn_error, vldt_error, err_array = Validate.Cross_validation_mxk(Net, model_params, criterion, optimizer, train_input, train_target, *crossval_params)



Vis.plot_interstates(err_array)

