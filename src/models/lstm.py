import torch
import torch.nn as nn
import torch.nn.functional as F
import math

############################################################################
# HS LSTM
############################################################################

class HSLSTMForDirection(nn.Module):
    def __init__(self, config, class_weights):
        super(HSLSTMForDirection, self).__init__()
        
        self.config = config
        model_params = config.model_params['hslstm']
        
        self.num_layers = model_params['num_layers']
        self.input_size = config.max_len
        self.hidden_size = self.input_size * 2
        self.dropout_prob = model_params['dropout_prob']
        self.output_size = min(len(class_weights), config.output_size)

        # Define model architecture
        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.input_size, self.hidden_size)])
        self.lstm_layers.extend([nn.LSTMCell(self.hidden_size, self.hidden_size) for _ in range(self.num_layers - 1)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout_prob) for _ in range(self.num_layers)])
        
        # Momentum predictor
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
        ## Loss functions
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.tensor(class_weights, dtype = torch.float32))
        
        self.device = self.config.device
        
    def forward(self, x, y = None):
        
        ### SHARED NETWORK
        h, c = [], []
        for _ in range(self.num_layers):
            h.append(torch.zeros(x.size(0), self.hidden_size).to(self.device))
            c.append(torch.zeros(x.size(0), self.hidden_size).to(self.device))
            
        for t in range(x.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    h[layer], c[layer] = self.lstm_layers[layer](x[:,t,:], (h[layer], c[layer]))
                else:
                    h[layer], c[layer] = self.lstm_layers[layer](h[layer-1], (h[layer], c[layer]))
                    
                if layer == self.num_layers - 1:
                    h[layer] = self.dropout_layers[layer](h[layer])
        
        out = self.fc(h[-1])
        
        if y is None:
            return out
        
        loss = self.loss_fn(out, y)
        return out, loss