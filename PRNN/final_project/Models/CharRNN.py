import torch
import torch.nn as nn
import torch.nn.functional as F

class CharRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden,n_output,n_layers,model,device):
        super(CharRNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.model = model
        self.encoder = nn.Linear(n_inputs, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_output)
        if self.model == "gru":
            self.rnn = nn.GRU(n_hidden, n_hidden, n_layers)
            self.hidden_init = nn.Linear(1,n_hidden)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(n_hidden, n_hidden, n_layers)
        self.device = device
        
    def forward(self, x):
        x=x.permute(1,0,2)
        hidden_dummy = torch.ones(self.n_layers,x.shape[1],1).to(self.device)
        outhist, hidden = self.rnn(self.encoder(x),self.hidden_init(hidden_dummy))
        output = self.decoder(outhist)
        
        return output, hidden