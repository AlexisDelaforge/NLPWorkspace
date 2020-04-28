import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# Not my code
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# Modifié

class EncoderRNN(nn.Module):
    def __init__(self, embedder, num_layers=1, bidirectional=False, device='cpu'):
        super(EncoderRNN, self).__init__()

        self.embedding = embedder
        self.hidden_size = self.embedding.embedding_dim
        self.input_size = len(self.embedding.word2index)
        self.device = device
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers, bidirectional=bidirectional).to(
            self.device)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.num_layers_X_directions = 2 * self.num_layers
        else:
            self.num_layers_X_directions = 1 * self.num_layers

    def forward(self, input, hidden):
        embedded = self.embedding(input)  # .view(1, 1, -1)
        output = embedded
        # print('Encoder zone')
        # print(input.shape)
        # print(hidden.shape)
        # print(output.shape)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers_X_directions, batch_size, self.hidden_size).to(self.device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, bidirectional=False, device='cpu'):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.num_layers_X_directions = 2 * self.num_layers
        else:
            self.num_layers_X_directions = 1 * self.num_layers

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers_X_directions, batch_size, self.hidden_size).to(self.device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, embedder, max_length, dropout_p=0.1, num_layers=1, bidirectional=False, device='cpu'):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = embedder.to(device)
        self.hidden_size = self.embedding.embedding_dim
        self.output_size = len(self.embedding.word2index)
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.num_layers_X_directions = 2 * self.num_layers
        else:
            self.num_layers_X_directions = 1 * self.num_layers

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input.to(torch.int64))
        embedded = self.dropout(embedded)
        # print('Decoder zone')
        # print(input.shape)
        # print(hidden.shape)
        # print(embedded.shape)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # print(output.shape)
        output = F.log_softmax(self.out(output[0]), dim=1)
        # print(output.shape)
        # output = self.out(output[0])
        # print('AttnDecoderRNN forward output')
        # print(output.shape)
        # print(hidden.shape)
        # print(attn_weights.shape)
        # print('Decoder output')
        # print(output.shape)
        # print(hidden.shape)
        # print(attn_weights.shape)
        return output, hidden, attn_weights

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers_X_directions, batch_size, self.hidden_size).to(self.device)


class AttnAutoEncoderRNN(nn.Module):
    def __init__(self, embedder, max_length, num_layers=1, bidirectional=False, dropout_p=0.1, device='cpu', teacher_forcing_ratio=0.5):
        super(AttnAutoEncoderRNN, self).__init__()
        self.max_length = max_length  # longueur maximum de la prediction
        self.device = device
        print(self.device)
        self.embedder = embedder
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_p = dropout_p
        self.encoder = EncoderRNN(embedder, num_layers=self.num_layers, bidirectional=self.bidirectional, device=self.device)
        self.decoder = AttnDecoderRNN(embedder, self.max_length, dropout_p=self.dropout_p, num_layers=self.num_layers, bidirectional=self.bidirectional, device=self.device)
        self.sos_token = self.embedder.get_index('<sos>')
        self.sos_token = torch.tensor(self.sos_token).to(self.device)
        self.sos_tensor = self.embedder(self.sos_token).to(self.device)
        # print('youpi')
        self.eos_token = self.embedder.get_index('<sos>')
        self.eos_token = torch.tensor(self.eos_token).to(self.device)
        self.eos_tensor = self.embedder(self.eos_token).to(self.device)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, source):

        # print(source)
        # print(source[1].shape)
        batch_size = source[0].size(1)  # Batch size
        # print('len source')
        # print(batch_size)
        # print('batch_size')
        # print(batch_size)
        input_tensor = source[0]  # .squeeze(0)
        target_tensor = source[1]  # .squeeze(0)
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_hidden = self.encoder.init_hidden(batch_size)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size).to(self.device)

        for ei in range(input_length):
            # print('shape premier batch de lettres')
            # print(input_tensor[ei].shape) # Batch X Embed Dim
            # print('shape encoder hidden')
            # print(input_tensor[ei].shape) # Batch X Hidden size
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei].unsqueeze(0).to(self.device), encoder_hidden)  # Voir pourquoi .unsqueeze(0)
            encoder_outputs[ei] = encoder_output[0, 0]
        # print(input_tensor[ei].unsqueeze(0).shape)
        # print(self.eos_tensor.shape)
        # print(eos_tensor.shape)
        # print(torch.stack([self.eos_tensor]*batch_size, dim=0).unsqueeze(0).shape)
        # print('last one')
        encoder_output, encoder_hidden = self.encoder(
            torch.stack([self.eos_token.to(torch.int64)] * batch_size, dim=0).unsqueeze(0).to(self.device),
            encoder_hidden)
        # print(torch.tensor(self.eos_token.repeat(0,batch_size)))
        encoder_outputs[ei + 1] = encoder_output[0, 0]

        # decoder_input = torch.tensor([[self.sos_token]]).to(self.device)
        decoder_input = torch.stack([self.sos_token.to(torch.int64)] * batch_size, dim=0).unsqueeze(0).to(self.device)
        # print('first target_tensor')
        # print(decoder_input.shape)
        # print(encoder_hidden.shape)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        # print(use_teacher_forcing)
        # print('a changer au dessus')
        decoder_output = None  # Simple declaration
        decoder_outputs = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                # print(di)
                # print('shape decod premier batch de lettres')
                # print(decoder_input.shape)  # Batch X Embed Dim
                # print('shape decod encoder hidden')
                # print(decoder_hidden.shape)  # Batch X Hidden size
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                # print('shape decod premier batch de lettres APRES')
                # print(decoder_input.shape)  # Batch X Embed Dim
                # print('shape decod encoder hidden APRES')
                # print(decoder_hidden.shape)  # Batch X Hidden size
                decoder_outputs.append(decoder_output)
                # print(decoder_hidden.shape)
                # print('target_tensor')
                # print(target_tensor.shape)
                # print(target_tensor[di].unsqueeze(0).shape)
                decoder_input = target_tensor[di].unsqueeze(0)  # Teacher forcing
                # print('decoder_input shape')
                # print(decoder_input.shape)
                if decoder_input.squeeze(0)[0].item() == self.eos_token:  # Test le premier (batch de même size)
                    # print('break')
                    break
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                # print('shape decod premier batch de lettres')
                # print(decoder_input.shape)  # Batch X Embed Dim
                # print('shape decod encoder hidden')
                # print(decoder_hidden.shape)  # Batch X Hidden size
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs.append(decoder_output)
                topv, topi = decoder_output.topk(1)
                # print(topv)
                # print(topi)
                decoder_input = topi.permute(1, 0).detach()  # detach from history as input
                # print('decoder_input shape')
                # print(decoder_input.shape)
                if decoder_input.squeeze(0)[0].item() == self.eos_token:  # Test le premier (batch de même size)
                    break

        return torch.stack(decoder_outputs, dim=0), target_tensor