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
        self.input_size = 1
        self.hidden_size = 32
        self.num_layers = 1
        self.output_size = 1
        self.dropout = 0

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        self.hidden_and_cell = None

        #######################################
        ### END OF YOUR CODE
        #######################################

    def forward(self, input):
        """
        Defines the forward pass of the module
        Args:
            input:     torch Variable (mini_batch_size x seq_len), of input pitch contours
        Returns:
            output: torch Variable (mini_batch_size), containing predicted ratings
        """
        #######################################
        ### BEGIN YOUR CODE HERE
        # implement the forward pass of model
        #######################################
        if torch.cuda.is_available():
            input = input.cuda()

        self.init_hidden(input.size()[0])
        input = torch.unsqueeze(torch.transpose(input, 0, 1), 2)

        lstm_out, self.hidden_and_cell = self.lstm(input, self.hidden_and_cell)
        output = self.linear(torch.sigmoid(lstm_out))

        #######################################
        ### END OF YOUR CODE
        #######################################
        return output

    def detach_hidden(self):
        """
        Detach hidden and cell state to not eat RAMs.
        """
        if self.hidden_and_cell:
            new_hidden = Variable(self.hidden_and_cell[0].data)
            new_cell = Variable(self.hidden_and_cell[1].data)
            if torch.cuda.is_available():
                new_hidden = new_hidden.cuda()
                new_cell = new_cell.cuda()
            self.hidden_and_cell = (new_hidden, new_cell)
        return    


    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the recurrent layers
        Args:
            mini_batch_size: number of data samples in the mini-batch
        """
        #######################################
        ### BEGIN YOUR CODE HERE
        #######################################
        hidden = torch.zeros(self.num_layers, mini_batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers, mini_batch_size, self.hidden_size)

        if torch.cuda.is_available():
            hidden = hidden.cuda()
            cell = cell.cuda()
        self.hidden_and_cell = (hidden, cell)

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
            input:     torch Variable (mini_batch_size x seq_len), of input pitch contours
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
