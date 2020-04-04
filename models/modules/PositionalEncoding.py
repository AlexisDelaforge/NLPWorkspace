import math
import torch
import torch.nn as nn

# Code from : https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# Change : No


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, device, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(p=dropout).to(self.device)
        pe = torch.zeros(max_len, d_model).to(self.device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(self.device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(self.device)
        pe[:, 0::2] = torch.sin(position * div_term).to(self.device)
        pe[:, 1::2] = torch.cos(position * div_term).to(self.device)
        pe = pe.unsqueeze(0).transpose(0, 1).to(self.device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x).to(self.device)

