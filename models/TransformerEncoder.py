import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import PositionalEncoding


# Code from : https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# Change : No


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, device='cpu'):
        super(TransformerModel, self).__init__()
        self.device = device
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, self.device, dropout).to(self.device)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout).to(self.device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers).to(self.device)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mode='train'):
        #print(src[0].shape)
        #print(src[1].shape)
        input = src[0].to(self.device)
        target = src[1].to(self.device)
        #print(input.device)
        #print(target.device)
        '''if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask'''
        input = self.pos_encoder(input)
        output = self.transformer_encoder(input, self.src_mask)
        output = self.decoder(output)
        #print(output.shape)
        return output.permute(0, 2, 1), target