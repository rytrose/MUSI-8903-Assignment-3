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
        super().__init__()
        #######################################
        ### BEGIN YOUR CODE HERE
        #######################################
        self.lstm_input_size = 1
        self.hidden_size = 256
        self.num_layers = 2
        self.output_size = 1
        self.dropout = 0

        self.avgpool = nn.AvgPool1d(10, padding=5)

        self.in_fc1 = nn.Linear(1, 32)
        self.in_fc2 = nn.Linear(32, 64)

        self.lstm = nn.LSTM(64, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)

        mid_point = (self.hidden_size + self.output_size) // 2
        self.out_fc1 = nn.Linear(self.hidden_size, mid_point)
        self.out_fc2 = nn.Linear(mid_point, self.output_size)

        self.hidden_and_cell = None
        self.init_params()

        #######################################
        ### END OF YOUR CODE
        #######################################

    def forward(self, seq):
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
            seq = seq.cuda()

        self.init_hidden(seq.size()[0])

        # maybe this isn't great to do, because you are reranging the data within a small section, while the 
        # validation will be over an entire sequence
        seq = 10*(seq - seq.mean())/seq.std()
        if len(seq.shape) < 2:
            seq = torch.unsqueeze(seq, 0)
        seq = torch.unsqueeze(seq, 1)
        seq = self.avgpool(seq)
        seq = torch.squeeze(seq)
        if len(seq.shape) < 2:
            seq = torch.unsqueeze(seq, 0)
        seq = torch.unsqueeze(seq, 2)
            
        seq = self.in_fc2(F.relu(self.in_fc1(seq)))
        lstm_out, self.hidden_and_cell = self.lstm(seq, self.hidden_and_cell)
        output = self.out_fc2(F.relu(self.out_fc1(lstm_out)))
    
        output = torch.squeeze(output[:, -1, :])
        #######################################
        ### END OF YOUR CODE
        #######################################
        return output

    def init_params(self):
         for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
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
        self.lstm_input_size = 1
        self.hidden_size = 128
        self.num_layers = 2
        self.output_size = 1
        self.dropout = 0

        self.conv1 = nn.Conv1d(1, 1, 30, padding=15)
        self.maxpool1 = nn.MaxPool1d(6, padding=3)
        self.conv2 = nn.Conv1d(1, 1, 15, padding=7)
        self.maxpool2 = nn.MaxPool1d(4, padding=2)

        self.in_fc1 = nn.Linear(1, 32)
        self.in_fc2 = nn.Linear(32, 64)
        self.lstm = nn.LSTM(64, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)

        mid_point = (self.hidden_size + self.output_size) // 2
        self.out_fc1 = nn.Linear(self.hidden_size, mid_point)
        self.out_fc2 = nn.Linear(mid_point, self.output_size)

        self.hidden_and_cell = None
        self.init_params()
        #######################################
        ### END OF YOUR CODE
        #######################################

    def forward(self, seq):
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
            seq = seq.cuda()

        self.init_hidden(seq.size()[0])

        # maybe this isn't great to do, because you are reranging the data within a small section, while the 
        # validation will be over an entire sequence
        seq = 10*(seq - seq.mean())/seq.std()
        if len(seq.shape) < 2:
            seq = torch.unsqueeze(seq, 0)
        seq = torch.unsqueeze(seq, 1)

        seq = self.maxpool1(F.relu(self.conv1(seq)))
        seq = self.maxpool2(F.relu(self.conv2(seq)))

        seq = torch.squeeze(seq, dim=1)
        seq = torch.unsqueeze(seq, 2)

        seq = self.in_fc2(F.relu(self.in_fc1(seq)))

        lstm_out, self.hidden_and_cell = self.lstm(seq, self.hidden_and_cell)
        output = self.out_fc2(F.relu(self.out_fc1(lstm_out)))
    
        output = torch.squeeze(output[:, -1, :])
        #######################################
        ### END OF YOUR CODE
        #######################################
        return output

        #######################################
        ### END OF YOUR CODE
        #######################################
        return output

    def init_params(self):
         for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
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
