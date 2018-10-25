import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PitchRnn(nn.Module):
    """
    Class to implement a deep RNN model for music performance assessment using
	 pitch contours as input
    """

    def __init__(self):
        """
        Initializes the PitchRnn class with internal parameters for the different layers
        This should be a RNN based model. You are free to choose your hyperparameters with 
        respect type of RNN cell (LSTM, GRU), type of regularization (dropout, batchnorm)
        and type of non-linearity
        """
        super(PitchRnn, self).__init__()
        #######################################
        ### BEGIN YOUR CODE HERE
        #######################################
        pass 
        #######################################
        ### END OF YOUR CODE
        #######################################

    def forward(self, input):
        """
        Defines the forward pass of the module
        Args:
            input: 	torch Variable (mini_batch_size x seq_len), of input pitch contours
        Returns:
            output: torch Variable (mini_batch_size), containing predicted ratings
        """
        #######################################
        ### BEGIN YOUR CODE HERE
        # implement the forward pass of model
        #######################################
        output = None
        #######################################
        ### END OF YOUR CODE
        #######################################
        return output

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the recurrent layers
        Args:
            mini_batch_size: number of data samples in the mini-batch
        """
        #######################################
        ### BEGIN YOUR CODE HERE
        #######################################
        pass 
        #######################################
        ### END OF YOUR CODE
        #######################################

class PitchCRnn(nn.Module):
    """
    Class to implement a deep CRNN model for music performance assessment using
	 pitch contours as input
    """

    def __init__(self):
        """
        Initializes the PitchCRnn class with internal parameters for the different layers
        This should be a RNN based model. You are free to choose your hyperparameters with 
        respect type of RNN cell (LSTM, GRU), type of regularization (dropout, batchnorm)
        and type of non-linearity. Since this is a CRNN model, you will also need to decide 
        how many convolutonal layers you want to take as input. Note that the minimum
        sequence lenth that you should expect is 2000. 
        """
        super(PitchCRnn, self).__init__()
        #######################################
        ### BEGIN YOUR CODE HERE
        #######################################
        pass 
        #######################################
        ### END OF YOUR CODE
        #######################################

    def forward(self, input):
        """
        Defines the forward pass of the module
        Args:
            input: 	torch Variable (mini_batch_size x seq_len), of input pitch contours
        Returns:
            output: torch Variable (mini_batch_size), containing predicted ratings
        """
        #######################################
        ### BEGIN YOUR CODE HERE
        # implement the forward pass of model
        #######################################
        output = None
        #######################################
        ### END OF YOUR CODE
        #######################################
        return output

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the recurrent layers
        Args:
            mini_batch_size: number of data samples in the mini-batch
        """
        #######################################
        ### BEGIN YOUR CODE HERE
        #######################################
        pass 
        #######################################
        ### END OF YOUR CODE
        #######################################