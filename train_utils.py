import os
import sys
import math
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import eval_utils

def augment_data(data):
    """
    Augments the data using pitch shifting
    Args:
        data: batched data
    """
    num_batches = len(data)
    aug_data = [None] * num_batches
    # create augmented data
    for batch_idx in range(num_batches):
        mini_batch_size, seq_len = data[batch_idx]['pitch_tensor'].size()
        pitch_shift = ((torch.rand(mini_batch_size, 1) * 4) - 2) / 72.0 # since we normalize between 36 to 108 MIDI Note
        pitch_shift = pitch_shift.expand(mini_batch_size, seq_len)
        pitch_tensor = data[batch_idx]['pitch_tensor'].clone()
        pitch_tensor[pitch_tensor != 0] = pitch_tensor[pitch_tensor != 0] + pitch_shift[pitch_tensor != 0]
        new_data = {}
        new_data['pitch_tensor'] = pitch_tensor
        new_data['score_tensor'] = data[batch_idx]['score_tensor'].clone()
        aug_data[batch_idx] = new_data
    # combine with orignal data
    aug_data = data + aug_data
    return aug_data

def train(model, criterion, optimizer, data, metric):
    """
    Returns the model performance metrics
    Args:
        model:          object, trained model of PitchContourAssessor class
        criterion:      object, of torch.nn.Functional class which defines the loss 
        optimizer:      object, of torch.optim class which defines the optimization algorithm
        data:           list, batched testing data
        metric:         int, from 0 to 3, which metric to evaluate against
    Returns:            (1,) torch Tensor, average MSE loss for all batches
    """
    # Put the model in training mode
    model.train() 
    # Initializations
    num_batches = len(data)
    loss_avg = 0
	# iterate over batches for training
    for batch_idx in range(num_batches):
		# clear gradients and loss
        model.zero_grad()
        loss = 0
        # extract pitch tensor and score for the batch
        pitch_tensor = data[batch_idx]['pitch_tensor']
        score_tensor = data[batch_idx]['score_tensor'][:, metric]
        
        #######################################
        ### BEGIN YOUR CODE HERE
        # perform the forward pass through the
        # network for a batch and compute the
        # average loss
        # perform the backward pass and the 
        # optimization step
        #######################################
        loss_avg = 0
        #######################################
        ### END OF YOUR CODE
        #######################################
        
    loss_avg /= num_batches
    return loss_avg

# define training and validate method
def train_and_validate(model, criterion, optimizer, train_data, val_data, metric):
    """
    Defines the training and validation cycle for the input batched data for the conv model
    Args:
        model:          object, trained model of PitchContourAssessor class
        criterion:      object, of torch.nn.Functional class which defines the loss 
        optimizer:      object, of torch.optim class which defines the optimization algorithm
        train_data:     list, batched training data
        val_data:       list, batched validation data
        metric:         int, from 0 to 3, which metric to evaluate against
    """
    # train the network
    train(model, criterion, optimizer, train_data, metric)   
    # evaluate the network on train data
    train_loss_avg, train_r_sq, train_accu= eval_utils.eval_model(model, criterion, train_data, metric)
    # evaluate the network on validation data
    val_loss_avg, val_r_sq, val_accu= eval_utils.eval_model(model, criterion, val_data, metric)
    # return values
    return train_loss_avg, train_r_sq, train_accu, val_loss_avg, val_r_sq, val_accu

def save(filename, perf_model, log_parameters=None):
    """
    Saves the saved model
    Args:
        filename:       name of the file 
        model:          torch.nn model 
        log_parameters: dict, contaning the log parameters
    """
    save_filename = 'saved/' + filename + '_Reg.pt'
    torch.save(perf_model.state_dict(), save_filename)
    if log_parameters is not None:
        log_filename = 'runs/' + filename + '_Log.txt'
        f = open(log_filename, 'w')
        f.write(str(log_parameters))
        f.close()
    print('Saved as %s' % save_filename)

def time_since(since):
    """
    Returns the time elapsed between now and 'since'
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def adjust_learning_rate(optimizer, epoch, adjust_every):
    """
    Adjusts the learning rate of the optimizer based on the epoch
    Args:
       optimizer:      object, of torch.optim class 
       epoch:          int, epoch number
       adjust_every:   int, number of epochs after which adjustment is to done
    """
    #######################################
    ### BEGIN YOUR CODE HERE
    # write your code to adjust the learning
    # rate based on the epoch number
    # You are free to implement your own 
    # method here
    #######################################
    pass 
    #######################################
    ### END OF YOUR CODE
    #######################################

def log_init():
    """
    Initializes the log element
    """
    log_parameters = {
        'x': [],
        'loss_train': [],
        'r_sq_train': [],
        'acc_train': [],
        'loss_val': [],
        'r_sq_val': [],
        'acc_val': [],
    }
    return log_parameters

def log_epoch_stats(
    log_parameters,
    epoch_index, 
    mean_loss_train,
    mean_rsq_train,
    mean_acc_train,
    mean_loss_val,
    mean_rsq_val,
    mean_acc_val
    ):
    """
    Logs the epoch statistics
    """
    log_parameters['x'].append(epoch_index)
    log_parameters['loss_train'].append(mean_loss_train)
    log_parameters['r_sq_train'].append(mean_rsq_train)
    log_parameters['acc_train'].append(mean_acc_train)
    log_parameters['loss_val'].append(mean_loss_val)
    log_parameters['r_sq_val'].append(mean_rsq_val)
    log_parameters['acc_val'].append(mean_acc_val)
    return log_parameters


