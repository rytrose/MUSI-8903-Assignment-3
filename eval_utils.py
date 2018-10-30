import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import r2_score

"""
Contains standard utility functions for training and testing evaluations
"""

def eval_regression(target, pred):
    """
    Calculates the standard regression evaluation  metrics
    Args:
        target:     (N x 1) torch Float tensor, actual ground truth
        pred:       (N x 1) torch Float tensor, predicted values from the regression model
    Returns:
        r_sq:       float, average r-squared metric
        accu:       float, average accuracy (between 0. to 1.)
    """
    #######################################
    ### BEGIN YOUR CODE HERE
    # write your code to compute the R-squared
    # metric and accuracy percentage
    # You may use packages such as scikitlearn
    # for this. 
    #######################################
    target_np = target.data.cpu().numpy()
    pred_np = pred.data.cpu().numpy()
    r_sq = r2_score(target_np, pred_np)
    rounded_pred = np.around(pred_np, decimals=1)
    num_equal = (target_np == rounded_pred).sum()
    accu = num_equal / pred_np.shape[0] 
    #######################################
    ### END OF YOUR CODE
    #######################################
    return r_sq, accu

def eval_model(model, criterion, data, metric, extra_outs=False):
    """
    Returns the model performance metrics
    Args:
        model:          object, trained model of PitchContourAssessor class
        criterion:      object, of torch.nn.Functional class which defines the loss 
        data:           list, batched testing data
        metric:         int, from 0 to 3, which metric to evaluate against
        extra_outs:     bool, returns the target and predicted values if true
    """
    # put the model in eval mode
    model.eval()
    # intialize variables
    num_batches = len(data)
    pred = np.array([])
    target = np.array([])
    loss_avg = 0
    # iterate over batches for validation
    for batch_idx in range(num_batches):
        # extract pitch tensor and score for the batch
        pitch_tensor = data[batch_idx]['pitch_tensor']
        score_tensor = data[batch_idx]['score_tensor'][:, metric]
        
        #######################################
        ### BEGIN YOUR CODE HERE
        # perform the forward pass through the
        # network for a batch and compute the
        # average loss
        # store the model output in 'model_output'
        #######################################
        output = model(pitch_tensor)

        if torch.cuda.is_available():
            score_tensor = score_tensor.cuda()

        loss = criterion(output, score_tensor)
        # print("predictions", output)
        # print("targets", score_tensor)

        loss_avg += loss.data
        model_output = output
        #######################################
        ### END OF YOUR CODE
        #######################################
        
        # concatenate target and pred for computing validation metrics
        pred = torch.cat((pred, model_output.data.view(-1)), 0) if pred.size else model_output.data.view(-1)
        target = torch.cat((target, score_tensor), 0) if target.size else score_tensor
    r_sq, accu = eval_regression(target, pred)
    loss_avg /= num_batches
    if extra_outs:
        return loss_avg, r_sq, accu, pred, target
    else:
        return loss_avg, r_sq, accu

