
import numpy as np
import Nets
from torch import nn
import torch
import Validate
import os

import dlc_bci as bci
train_input , train_target = bci.load(root = './ data_bci')
print ('train_input:', str ( type ( train_input ) ) , train_input.size () )
print ('train_target:', str ( type ( train_target ) ) , train_target.size () )
test_input , test_target = bci.load ( root = './data˙bci' , train = False )
print ('test_input:', str ( type ( test_input ) ) , test_input.size () )
print ('test target:', str ( type ( test_target ) ) , test_target.size () )

# Use cuda if available
if torch.cuda.is_available():
    train_input, train_target = train_input.cuda(), train_target.cuda()
    test_input, test_target = test_input.cuda(), test_target.cuda()

size = train_input.size() 


def sweep_first():
    Net = Nets.Net1             #Net1 as defined in Nets.py has just on linear hidden layer
    
    m_fold = 2                  #number of folds of full k_fold cross validations
    k_fold = 5     
    nb_hidden = 50            #number of folds of the cross-validation
    nb_iters = 2500           #(max) number of iterations during training
    batch_sizes = [1, 5, 50, 316]         #size of the batches during training
         
    use_tol = True             #use the tolerance as stopping criterion or not. train_error is used, so inter_states = True is required
    inter_states = True         #compute the train and test errors every 25 iterations, print them and save them in err_array
    
    err_list = [0,]*len(batch_sizes)
    
    for i_, batch_size in enumerate(batch_sizes):
        print('##### Testing batch size: {} #####'.format(batch_size))
        model_params = [size, nb_hidden]
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD
        crossval_params = [m_fold, k_fold, nb_iters, batch_size, use_tol, inter_states]
        #Apply crossvalidation
        trn_error, vldt_error, err_array = Validate.Cross_validation_mxk(Net, model_params, criterion, optimizer, train_input, train_target, *crossval_params)
        
        err_list[i_] = err_array
    
    np.save(os.path.join('arrays_and_images','first_err_list'),err_list)
    

    
def sweep_second():            
    Net = Nets.Net1             #Net1 as defined in Nets.py has just on linear hidden layer
    nb_hiddens = [5,20,100,500]
    
    m_fold = 2                  #number of folds of full k_fold cross validations
    k_fold = 5                 #number of folds of the cross-validation
    nb_iters = 400             #(max) number of iterations during training
    batch_size = 5      #size of the batches during training
    use_tol = False             #use the tolerance as stopping criterion or not  
    inter_states = True         #compute the train and test errors every 25 iterations, print them and save them in err_array
    
    crossval_params = [m_fold, k_fold, nb_iters, batch_size, use_tol, inter_states]
    err_list = [0,]*len(nb_hiddens)
    
    for i_,nb_hidden in enumerate(nb_hiddens):
        print('##### Testing number of hidden neurons: {} #####'.format(nb_hidden))
        model_params = [size, nb_hidden]
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD
        
        #Apply crossvalidation
        trn_error, vldt_error, err_array = Validate.Cross_validation_mxk(Net, model_params, criterion, optimizer, train_input, train_target, *crossval_params)
        
        err_list[i_] = err_array
    
    np.save(os.path.join('arrays_and_images','second_err_list'),err_list)
    
    
def sweep_third():
    Net = Nets.Net2            #Net1 as defined in Nets.py has just on linear hidden layer
    nb_hiddens = [5,20,100,500]
    
    m_fold = 2                  #number of folds of full k_fold cross validations
    k_fold = 5                 #number of folds of the cross-validation
    nb_iters = 300             #(max) number of iterations during training
    batch_size = 5      #size of the batches during training
    use_tol = True             #use the tolerance as stopping criterion or not         
    inter_states = True         #compute the train and test errors every 25 iterations, print them and save them in err_array
    
    crossval_params = [m_fold, k_fold, nb_iters, batch_size, use_tol, inter_states]
    err_list = [0,]*len(nb_hiddens)
    
    for i_,nb_hidden in enumerate(nb_hiddens):
        print('##### Testing number of hidden neurons: {} #####'.format(nb_hidden))
        model_params = [size, nb_hidden]
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam
        
        #Apply crossvalidation
        trn_error, vldt_error, err_array = Validate.Cross_validation_mxk(Net, model_params, criterion, optimizer, train_input, train_target, *crossval_params)
        
        err_list[i_] = err_array
    
    np.save(os.path.join('arrays_and_images','third_err_list'),err_list)

    
#sweep_third()

    