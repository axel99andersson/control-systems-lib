import torch
import torch.nn as nn

class LSTMRecoveryController(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(LSTMRecoveryController, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,dropout=dropout, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=hidden_size // 2)
        self.fc2 = nn.Linear(in_features=hidden_size // 2, out_features=1)

    def forward(self, x):
        out, (last_h_state, last_c_state) = self.lstm(x)
        last_h_state = last_h_state[-1,:,:]
        x = self.fc1(last_h_state)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x
