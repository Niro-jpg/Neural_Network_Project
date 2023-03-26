import torch
from torch import nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size = 256, batch_size = 32):
        super(RNN, self).__init__()

        self.batch_size = batch_size
        
        self.kwargs = {'input_size': input_size, 'hidden_size': hidden_size, 'batch_size': batch_size}

        # Defining some parameters
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input, hidden = None):

        input = input.permute(1,0,2)

        if hidden == None: hidden = self.init_hidden(input.size()[1])

        for element in input:
            combined = torch.cat((element, hidden), 1)
            hidden = self.i2h(combined)

        output = self.sigmoid(self.h2o(hidden))

        return output, hidden
    
    def init_hidden(self, batch_size = None):

        if batch_size == None: batch_size = self.batch_size

        hidden = torch.zeros(batch_size, self.hidden_size)
        return hidden
    

class SRNN(nn.Module):
    def __init__(self, input_size, hidden_size = 256, MLP_len = 3, batch_size = 32, shift = 1):
        super(SRNN, self).__init__()
        
        self.kwargs = {'input_size': input_size, 'hidden_size': hidden_size, 'batch_size': batch_size, 'shift': shift}
        
        self.shift = shift

        # Defining some parameters
        self.MLP_len = MLP_len

        self.hidden_size = hidden_size

        self.h2o = nn.Linear(hidden_size, 1, dtype = torch.float)
        self.sigmoid = nn.Sigmoid()

        self.batch_size = batch_size

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

        input = input.permute(1,0,2)
        
        if hidden == None: hidden = self.init_hidden(input.size()[1])

        for x in input:

            pre_hidden = torch.roll(hidden, self.shift, -1)
            linear_x = self.linearX(x)
            fx = self.layerMLP(x)
            b = torch.mul(fx,torch.sigmoid(linear_x))
            hidden = F.relu(pre_hidden + b)

        output = self.sigmoid(self.h2o(hidden))

        return output, hidden
    
    def init_hidden(self, batch_size = None):

        if batch_size == None: batch_size = self.batch_size

        hidden = torch.zeros(batch_size, self.hidden_size)
        return hidden
    
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size = 256, batch_size = 32):
        super(GRU, self).__init__()
        
        self.kwargs = {'input_size': input_size, 'hidden_size': hidden_size, 'batch_size': batch_size}

        self.batch_size = batch_size

        # Defining some parameters
        self.hidden_size = hidden_size
        self.c2r = nn.Linear(input_size + hidden_size, hidden_size)
        self.c2z = nn.Linear(input_size + hidden_size, hidden_size)
        self.c2t = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, 1)
    
    def forward(self, input, hidden = None):
        
        input = input.permute(1,0,2)
        
        if hidden == None: hidden = self.init_hidden(input.size()[1])

        for element in input:

            combined = torch.cat((element, hidden), 1)

            r = torch.sigmoid(self.c2r(combined))

            z = torch.sigmoid(self.c2z(combined))

            combined_tilde = torch.cat((element, torch.mul(r,hidden)),1)

            hidden_tilde = torch.tanh(self.c2t(combined_tilde))

            hidden = torch.mul(z, hidden) + torch.mul(1 - z, hidden_tilde)

        output = self.sigmoid(self.h2o(hidden))

        return output, hidden
    
    def init_hidden(self, batch_size = None):

        if batch_size == None: batch_size = self.batch_size

        hidden = torch.zeros(batch_size, self.hidden_size)
        return hidden
    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size = 256, batch_size = 32):
        super(LSTM, self).__init__()
        
        self.kwargs = {'input_size': input_size, 'hidden_size': hidden_size, 'batch_size': batch_size}

        self.batch_size = batch_size

        # Defining some parameters
        self.hidden_size = hidden_size
        self.c2i = nn.Linear(input_size + hidden_size, hidden_size)
        self.c2f = nn.Linear(input_size + hidden_size, hidden_size)
        self.c2ot = nn.Linear(input_size + hidden_size, hidden_size)
        self.c2ct = nn.Linear(input_size + hidden_size, hidden_size)
        self.c22o = nn.Linear(hidden_size*2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input, hidden = None, covariate = None):
        
        input = input.permute(1,0,2)
        
        if hidden == None: hidden = self.init_hidden(input.size()[1])

        if covariate == None: covariate = self.init_covariate(input.size()[1])

        for element in input:

            combined = torch.cat((element, hidden), 1)

            i = torch.sigmoid(self.c2i(combined))

            f = torch.sigmoid(self.c2f(combined))

            o = torch.sigmoid(self.c2ot(combined))

            c_tilde = torch.tanh(self.c2ct(combined))

            covariate = torch.mul(f,covariate) + torch.mul(i,c_tilde)

            hidden = torch.mul(o,torch.tanh(covariate))

        output = self.sigmoid(self.c22o(torch.cat((hidden, covariate), 1)))

        return output, hidden
    
    def init_hidden(self, batch_size = None):

        if batch_size == None: batch_size = self.batch_size

        hidden = torch.zeros(batch_size, self.hidden_size)
        return hidden 
    
    def init_covariate(self, batch_size = None):

        if batch_size == None: batch_size = self.batch_size

        covariate = torch.zeros(batch_size, self.hidden_size)
        return covariate

        

class Net2(nn.Module):
    def __init__(self, input_size,hidden_size = 256, batch_size = 32):
        super(Net2, self).__init__()

        self.hidden_size = hidden_size

        self.batch_size = batch_size

        self.GRU = GRU(input_size,1, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden = None, covariate = None):

        output, hidden = self.GRU.forward(input)
        return self.sigmoid(output), hidden

    def init_hidden(self, batch_size = None):

        if batch_size == None: batch_size = self.batch_size

        hidden = torch.zeros(batch_size,self.hidden_size)
        return hidden    

class Net3(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 256, batch_size = 32):
        super(Net3, self).__init__()

        self.hidden_size = hidden_size

        self.batch_size = batch_size

        self.GRU = nn.GRU(input_size, hidden_size,1)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden = None, covariate = None):

        input = input.permute(1,0,2)
        
        if hidden == None: hidden = self.init_hidden(input.size()[1])

        _, hidden = self.GRU.forward(input)

        output = self.sigmoid(self.fc(hidden))

        return output, hidden

    def init_hidden(self, batch_size = None):

        if batch_size == None: batch_size = self.batch_size

        hidden = torch.zeros(batch_size, self.hidden_size)
        return hidden                  