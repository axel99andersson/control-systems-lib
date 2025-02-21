import torch
import torch.nn as nn

class LSTMRecoveryController(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, seq_len):
        super(LSTMRecoveryController, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,dropout=dropout, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=hidden_size // 2)
        self.fc2 = nn.Linear(in_features=hidden_size // 2, out_features=1)

    def forward(self, x):
        out, (last_h_state, last_c_state) = self.lstm(x)
        out = self.sigmoid(out)
        out = self.fc1(out)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
    