import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Not my code
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# Modifié

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, device='cpu'):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional).to(self.device)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.num_layers_X_directions = 2*self.num_layers
        else:
            self.num_layers_X_directions = 1 * self.num_layers

    def forward(self, input, hidden):
        # embedded = self.embedding(input).view(1, 1, -1)
        # output = embedded
        #print(input.shape)
        #print(hidden.shape)
        output, hidden = self.gru(input, hidden)
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
            self.num_layers_X_directions = 2*self.num_layers
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
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1, num_layers=1, bidirectional=False, device='cpu'):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.num_layers_X_directions = 2*self.num_layers
        else:
            self.num_layers_X_directions = 1 * self.num_layers

    def forward(self, input, hidden, encoder_outputs):
        #print('Input AttnDecoderRNN')
        #print(input.shape)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        #print('Embedded AttnDecoderRNN')
        #print(embedded.shape)
        #print('hidden AttnDecoderRNN')
        #print(hidden.shape)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        #print(output.shape)
        output = F.log_softmax(self.out(output[0]), dim=1)
        #print(output.shape)
        #output = self.out(output[0])
        #print('AttnDecoderRNN forward output')
        #print(output.shape)
        #print(hidden.shape)
        #print(attn_weights.shape)
        return output, hidden, attn_weights

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers_X_directions, batch_size, self.hidden_size).to(self.device)


class AttnAutoEncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, sos_token, eos_token, max_length, dropout_p=0.1, device='cpu', teacher_forcing_ratio = 0.5):
        super(AttnAutoEncoderRNN, self).__init__()
        self.max_length = max_length  # longueur maximum de la prediction
        self.device = device
        print(self.device)
        self.encoder = EncoderRNN(vocab_size, embed_size, device=self.device)
        self.decoder = AttnDecoderRNN(embed_size, vocab_size, self.max_length, device=self.device)
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, source):

        #print(source[0].shape)
        #print(source[1].shape)
        batch_size = source[0].size(1) # Batch size
        #print('batch_size')
        #print(batch_size)
        input_tensor = source[0] #.squeeze(0)
        target_tensor = source[1] #.squeeze(0)
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
                input_tensor[ei].unsqueeze(0).to(self.device), encoder_hidden) # Voir pourquoi .unsqueeze(0)
            encoder_outputs[ei] = encoder_output[0, 0]
        print(input_tensor[ei].unsqueeze(0).shape)
        print(torch.tensor(self.eos_token.repeat(0, batch_size)).unsqueeze(0).shape)
        encoder_output, encoder_hidden = self.encoder(
            torch.tensor(self.eos_token).to(self.device), encoder_hidden)
        print(torch.tensor(self.eos_token.repeat(0,batch_size)))
        encoder_outputs[ei+1] = encoder_output[0, 0]

        # decoder_input = torch.tensor([[self.sos_token]]).to(self.device)
        decoder_input = torch.tensor([[self.sos_token]*batch_size]).to(self.device)
        #print('first target_tensor')
        #print(decoder_input.shape)
        #print(encoder_hidden.shape)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True # if random.random() < self.teacher_forcing_ratio else False
        #print('a changer au dessus')
        decoder_output = None  # Simple declaration
        decoder_outputs = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                #print(di)
                #print('shape decod premier batch de lettres')
                #print(decoder_input.shape)  # Batch X Embed Dim
                #print('shape decod encoder hidden')
                #print(decoder_hidden.shape)  # Batch X Hidden size
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs.append(decoder_output)
                #print(decoder_hidden.shape)
                #print('target_tensor')
                #print(target_tensor.shape)
                #print(target_tensor[di].unsqueeze(0).shape)
                decoder_input = target_tensor[di].unsqueeze(0)  # Teacher forcing

                if decoder_input.item() == self.eos_token:
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
                print(topv)
                print(topi)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                if decoder_input.item() == self.eos_token:
                    break

        return torch.stack(decoder_outputs, dim=0), target_tensor