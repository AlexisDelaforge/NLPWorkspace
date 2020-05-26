import torch
import torch.nn as nn
import torch.nn.functional as F
import training_functions
import random


# Not my code
# https://www.kaggle.com/yokolet/quora-sentiment-analysis-by-pytorch
# Non modifié

# Only LSTM (2 or 3 layers) model suffered an overfitting problem.
# To avoid the problem, GRU and average pooling layer were added.
# The overfitting got better, but still the problem exists.

class SentimentRNN(nn.Module):
    def __init__(self, n_out, n_hidden, n_layers, embedder,
                 bidirectional=False, dropout=0.5, layer_dropout=0.3, device='cpu'):
        super(SentimentRNN, self).__init__()

        self.n_out = n_out  # num classes
        self.n_hidden = n_hidden  # num_hidden in cells
        self.n_layers = n_layers  # num_layers in cells
        self.embedding = embedder
        self.device = device
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1

        num_embeddings, embedding_dim = self.embedding.weights.shape

        # embedding layer
        # self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # self.embedding.weight.data.copy_(torch.from_numpy(weights)) # a voir pour le weights
        # self.embedding.weight.requires_grad = False # a voir pour le non update des weights
        # for some reason from_pretrained doesn't work
        # self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(weights))

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, n_hidden, n_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        # GRU layer
        self.gru = nn.GRU(embedding_dim, n_hidden, n_layers,
                          batch_first=True, dropout=dropout,
                          bidirectional=bidirectional)
        # Conv1d layer
        self.conv1d = nn.Conv1d(n_hidden * self.direction, (n_hidden * self.direction) // 2, 1)
        # Average Pooling layer
        self.avp = nn.AvgPool1d(2)
        # Dropout layer
        self.dropout = nn.Dropout(layer_dropout)
        # Fully-conneted layer
        self.fc = nn.Linear((n_hidden * self.direction) // 4 * 2, n_out)

        # Sigmoid activation layer
        self.sig = nn.Sigmoid()

    def forward(self, source, need_embedding=True):

        # source tensor à traiter de la même manière que dans Seq2SeqRNN

        batch_size = source[0].size(1)  # Batch size
        input_tensor = source[0]  # .squeeze(0)
        target_tensor = source[1]  # .squeeze(0)
        seq_len = input_tensor.size(0) # sentences length
        # target_length = target_tensor.size(0) # sentences length
        lstm_hidden, gru_hidden = self.init_hidden(batch_size)

        # simple declaration, self.max_length+1 gérer plus longue phrases + EOS
        # encoder_outputs = torch.zeros(self.max_length+1, self.encoder.encode_size).to(self.device)
        for ei in range(seq_len):

            if need_embedding:
                embeds = self.embedding(input_tensor[ei]).unsqueeze(1)
            else :
                embeds = input_tensor[ei].unsqueeze(1)
            # print(embeds.shape)
            # print(lstm_hidden[0].shape)
            # print(lstm_hidden[1].shape)
            lstm_out, lstm_hidden = self.lstm(embeds, lstm_hidden) # num_layers * num_directions X batch X hidden_size
            # print(lstm_out.shape)
            # print(lstm_hidden[0].shape)
            # print(lstm_hidden[1].shape)
            lstm_out = lstm_out.contiguous().view(-1, self.n_hidden * self.direction, 1)
            # print('before conv')
            # print(lstm_out.shape)
            lstm_out = self.conv1d(lstm_out)
            lstm_out = lstm_out.contiguous().view(-1, 1, (self.n_hidden * self.direction) // 2)
            lstm_out = self.avp(lstm_out)

            gru_out, gru_hidden = self.gru(embeds, gru_hidden)
            gru_out = gru_out.contiguous().view(-1, self.n_hidden * self.direction, 1)
            gru_out = self.conv1d(gru_out)
            gru_out = gru_out.contiguous().view(-1, 1, (self.n_hidden * self.direction) // 2)
            gru_out = self.avp(gru_out)

            # out = (lstm_out + gru_out) / 2.0
            out = torch.cat((lstm_out, gru_out), 2)
            out = self.dropout(out)
            out = self.dropout(out)
            out = self.fc(out.float())
            sig_out = self.sig(out)
            # print('value sig_out')
            # print(sig_out)
            sig_out = sig_out.view(batch_size, -1)
            # sig_out = sig_out[:, -1]  # get only last labels
            # print(sig_out)

        return sig_out, (lstm_hidden, gru_hidden)

    def init_hidden(self, batch_size, bidirectional=False):
        weight = next(self.parameters()).data
        # for LSTM (initial_hidden_state, initial_cell_state)
        lstm_hidden = (
            weight.new(self.n_layers * self.direction, batch_size, self.n_hidden).zero_().to(self.device),
            weight.new(self.n_layers * self.direction, batch_size, self.n_hidden).zero_().to(self.device)
        )
        # for GRU, initial_hidden_state
        gru_hidden = weight.new(self.n_layers * self.direction, batch_size, self.n_hidden).zero_().to(self.device)
        return lstm_hidden, gru_hidden
