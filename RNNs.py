import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 256):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden = None):

        if hidden == None: hidden = self.init_hidden()

        for element in input:

          combined = torch.cat((element, hidden), 0)

          hidden = self.i2h(combined)
          output = self.h2o(hidden)

        output = self.h2o(hidden)

        return output, hidden
    
    def init_hidden(self):

        hidden = torch.zeros(self.hidden_size)
        return hidden
    

class SRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 256, MLP_len = 3):
        super(SRNN, self).__init__()

        # Defining some parameters
        self.MLP_len = MLP_len

        self.hidden_size = hidden_size

        self.h2o = nn.Linear(hidden_size, output_size, dtype = torch.float)

        self.layerMLP = nn.Sequential(
          nn.Linear(input_size, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 50),
          nn.ReLU(),
          nn.Linear(50, hidden_size),
          nn.Sigmoid()
        )


        self.linearH = nn.Linear(hidden_size, hidden_size, bias = False, dtype = torch.float)
        
        self.linearX = nn.Linear(input_size, hidden_size, dtype = torch.float)
        
    
    def forward(self, input, hidden = None):
        
        if hidden == None: hidden = self.init_hidden()

        for x in input:

          pre_hidden = torch.roll(hidden, 1, -1)

          linear_x = self.linearX(x)

          fx = self.layerMLP(x)

          b = torch.mul(fx,torch.sigmoid(linear_x))

          hidden = F.relu(pre_hidden + b)

        output = self.h2o(hidden)

        return output, hidden
    
    def init_hidden(self):

        hidden = torch.zeros(self.hidden_size)
        return hidden
    
    
class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 256):
        super(GRU, self).__init__()

        # Defining some parameters
        self.hidden_size = hidden_size
        self.c2r = nn.Linear(input_size + hidden_size, hidden_size)
        self.c2z = nn.Linear(input_size + hidden_size, hidden_size)
        self.c2t = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden = None):
        
        if hidden == None: hidden = self.init_hidden()

        for element in input:

            combined = torch.cat((element, hidden), 0)

            r = torch.sigmoid(self.c2r(combined))

            z = torch.sigmoid(self.c2z(combined))

            combined_tilde = torch.cat((element, torch.mul(r,hidden)),0)

            hidden_tilde = torch.tanh(self.c2t(combined_tilde))

            hidden = torch.mul(z, hidden) + torch.mul(1 - z, hidden_tilde)

        output = self.h2o(hidden)

        return output, hidden
    
    def init_hidden(self):

        hidden = torch.zeros(self.hidden_size)
        return hidden    
    

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 256):
        super(LSTM, self).__init__()

        # Defining some parameters
        self.hidden_size = hidden_size
        self.c2i = nn.Linear(input_size + hidden_size, hidden_size)
        self.c2f = nn.Linear(input_size + hidden_size, hidden_size)
        self.c2ot = nn.Linear(input_size + hidden_size, hidden_size)
        self.c2ct = nn.Linear(input_size + hidden_size, hidden_size)
        self.c22o = nn.Linear(hidden_size*2, output_size)
    
    def forward(self, input, hidden = None, covariate = None):
        
        if hidden == None: hidden = self.init_hidden()

        if covariate == None: covariate = self.init_covariate()

        for element in input:

            combined = torch.cat((element, hidden), 0)

            i = torch.sigmoid(self.c2i(combined))

            f = torch.sigmoid(self.c2f(combined))

            o = torch.sigmoid(self.c2ot(combined))

            c_tilde = torch.tanh(self.c2ct(combined))

            covariate = torch.mul(f,covariate) + torch.mul(i,c_tilde)

            hidden = torch.mul(o,torch.tanh(covariate))

        output = self.c22o(torch.cat((hidden, covariate), 0))

        return output, hidden
    
    def init_hidden(self):

        hidden = torch.zeros(self.hidden_size)
        return hidden    
    
    def init_covariate(self):

        covariate = torch.zeros(self.hidden_size)
        return covariate
    
class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 256):
        super(Net, self).__init__()

        self.hidden_size = hidden_size

        self.SRNN = SRNN(input_size, output_size, hidden_size)

        self.layerMLP = nn.Sequential(
          nn.Linear(self.hidden_size, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 50),
          nn.Linear(50, output_size),
        )    

    def forward(self, input, hidden = None, covariate = None):
        
        if hidden == None: hidden = self.init_hidden()

        _, hidden = self.SRNN.forward(input, hidden)

        output = self.layerMLP(hidden)

        return output, hidden

    def init_hidden(self):

        hidden = torch.zeros(self.hidden_size)
        return hidden    
        