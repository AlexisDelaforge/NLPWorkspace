import torch
import torch.nn as nn
import torch.nn.functional as F
import training_functions
import random


# Not my code
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# Largement modifié

class EncoderRNN(nn.Module):
    def __init__(self, embedder, num_layers=1, bidirectional=False, encode_size=300, device='cpu'):
        super(EncoderRNN, self).__init__()

        self.embedding = embedder # Embbeder
        self.hidden_size = self.embedding.embedding_dim # Embbeding dimension
        self.input_size = len(self.embedding.word2index) # Vocabulary size
        self.device = device
        # self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.encode_size = encode_size # Real Hidden size
        self.gru = nn.GRU(self.hidden_size, self.encode_size, num_layers=num_layers, bidirectional=bidirectional).to(
            self.device) # Encoder GRU
        self.num_layers = num_layers # Num layers in GRU cells
        self.bidirectional = bidirectional # If the GRU is bidirectionnal (always false)
        if self.bidirectional:
            self.num_layers_X_directions = 2 * self.num_layers
        else:
            self.num_layers_X_directions = 1 * self.num_layers

    def forward(self, input, hidden, need_embedding=True):
        # print(input)
        if need_embedding:
            output = self.embedding(input.long())  # .view(1, 1, -1) # Embbed
        else:
            output = input

        # output shape :
        # print('Encoder zone')
        # print(training_functions.sentences_idx_to_word(input, self.embedding.word2index))
        # print(input.shape)
        # print(hidden.shape)
        # print(output.shape)

        output, hidden = self.gru(output, hidden) # Input go through the encoder GRU celle
        # output # sequence_size (always 1) X batch_size X encode_size
        # hidden # num_layers_*_directions X batch_size X encode_size

        # print('Encoder zone output')
        # print(output.shape)
        # print(hidden.shape)

        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers_X_directions, batch_size, self.encode_size).to(self.device)


class DecoderRNN(nn.Module): # Useless actually
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
    def __init__(self, embedder, max_length, dropout_p=0.1, num_layers=1, bidirectional=False, encode_size = 300, device='cpu'):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = embedder.to(device) # embbeder
        self.hidden_size = self.embedding.embedding_dim # embedding size
        self.output_size = len(self.embedding.word2index) # vocabulary size
        self.dropout_p = dropout_p # probality of dropout
        self.max_length = max_length # longest sentence
        self.device = device
        self.encode_size = encode_size # hidden size of GRU cells
        # self.attn = nn.Linear(self.hidden_size * 2, self.max_length+1) # Gestion plus longue phrase
        self.attn = nn.Linear(self.hidden_size + self.encode_size, self.max_length+1) # Gestion plus longue phrase
        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size + self.encode_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p) # Dropout layer
        self.gru = nn.GRU(self.hidden_size, self.encode_size, num_layers=num_layers, bidirectional=bidirectional) # Decoder GRU Cell
        self.out = nn.Linear(self.encode_size, self.output_size) # Fully connected final for code to vocabulary
        self.num_layers = num_layers # numlayers of GRU cells
        self.bidirectional = bidirectional # always false
        if self.bidirectional:
            self.num_layers_X_directions = 2 * self.num_layers
        else:
            self.num_layers_X_directions = 1 * self.num_layers

    def forward(self, input, hidden, encoder_outputs):
        # input shape : sequence_len (+1 each iter) x batch_size
        # hidden : numlayers X batch_size X encode_size
        # encoder_outputs : batch_size X encode_size /!\ C'est le code qui représente une phrase

        embedded = self.embedding(input.to(torch.int64)) # sequence_len (+1 each iter) x batch_size X embed_dim
        embedded = self.dropout(embedded) # sequence_len (+1 each iter) x batch_size X embed_dim

        # print('Decoder zone')
        # print(input.shape)
        # print(hidden.shape)
        # print(embedded.shape)
        # print(torch.cat((embedded[0], hidden[0]), 1).shape)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        # attn_weights.unsqueeze(0) : 1 (unsqueeze) X Bath_size X max_sequence_len

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # attn_applied.shape :  1 (unsqueeze) X Bath_size X encode_size
        # /!\ C'est le code qui représente une phrase

        output = torch.cat((embedded[-1], attn_applied[0]), 1) # Bath_size X encode_size+embbed_size (last word embedd)

        output = self.attn_combine(output).unsqueeze(0) # 1 X Batch_size X embbed_size

        # print(output.shape)

        output = F.relu(output) # 1 X Batch_size X embbed_size

        # print('c la wax')
        # print(output.shape)
        # print('c la wax 2')
        # print(hidden.shape)

        output, hidden = self.gru(output, hidden)
        # output # sequence_size (always 1) X batch_size X encode_size
        # hidden # num_layers_*_directions X batch_size X encode_size

        output = F.log_softmax(self.out(output[0]), dim=1)  # batch_size X vocab_size

        # print(output.topk(1)[1])
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
    def __init__(self, embedder, max_length, num_layers=1, bidirectional=False, encode_size= 300, dropout_p=0.1, device='cpu', teacher_forcing_ratio=0.5):
        super(AttnAutoEncoderRNN, self).__init__()
        self.max_length = max_length  # longueur maximum de la prediction
        self.device = device
        print(self.device)
        self.embedder = embedder # embedder
        self.num_layers = num_layers # num layers in GRU cells
        self.bidirectional = bidirectional # always false
        self.dropout_p = dropout_p # dropout prob
        self.encode_size = encode_size # encode size / hidden GRU cells /
        self.encoder = EncoderRNN(self.embedder, num_layers=self.num_layers, bidirectional=self.bidirectional, encode_size=self.encode_size, device=self.device)
        self.decoder = AttnDecoderRNN(self.embedder, self.max_length, dropout_p=self.dropout_p, num_layers=self.num_layers, bidirectional=self.bidirectional, encode_size=self.encode_size, device=self.device)
        self.sos_token = self.embedder.get_index('<sos>') # start of sentence
        self.sos_token = torch.tensor(self.sos_token).to(self.device)
        self.sos_tensor = self.embedder(self.sos_token).to(self.device)
        # print('youpi')
        self.eos_token = self.embedder.get_index('<eos>') # end of sentence
        self.eos_token = torch.tensor(self.eos_token).to(self.device)
        self.eos_tensor = self.embedder(self.eos_token).to(self.device)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def encode(self, input_tensor, batch_size, need_embedding):

        # print(source)
        # print(source[1].shape)
        # print('len source')
        # print(batch_size)
        # print('batch_size')
        # print(batch_size)
        input_length = input_tensor.size(0) # sentences length
        encoder_hidden = self.encoder.init_hidden(batch_size)

        # simple declaration, self.max_length+1 gérer plus longue phrases + EOS
        encoder_outputs = torch.zeros(self.max_length+1, self.encoder.encode_size).to(self.device)
        # print(encoder_outputs.shape)
        # print('input_length l183')
        # print(input_length)
        # print(input_tensor.shape)
        for ei in range(input_length):
            # print('shape premier batch de lettres')
            # print(input_tensor[ei].shape) # Batch X Embed Dim
            # print('shape encoder hidden')
            # print(input_tensor[ei].shape) # Batch X Hidden size
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei].unsqueeze(0).to(self.device), encoder_hidden, need_embedding)  # Voir pourquoi .unsqueeze(0)
            # print(encoder_outputs.shape)
            # print(encoder_output.shape)
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

        return encoder_hidden, encoder_outputs

    def decode(self, encoder_hidden, encoder_outputs, batch_size, target_tensor, use_teacher_forcing=None):
        target_length = target_tensor.size(0)
        # decoder_input = torch.tensor([[self.sos_token]]).to(self.device)
        decoder_input = torch.stack([self.sos_token.to(torch.int64)] * batch_size, dim=0).unsqueeze(0).to(self.device)
        # print('first target_tensor')
        # print(decoder_input.shape)
        # print(encoder_hidden.shape)
        decoder_hidden = encoder_hidden

        if use_teacher_forcing is not None:
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        # print(use_teacher_forcing)
        # print('a changer au dessus')
        decoder_output = None  # Simple declaration
        decoder_outputs = []
        # print(encoder_outputs.shape)
        # encoder_outputs = torch.ones_like(encoder_outputs)
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
                # print(decoder_input.type())
                # print(target_tensor[di].unsqueeze(0).type())
                decoder_input = torch.cat([decoder_input, target_tensor[di].unsqueeze(0)], dim=0)  # Teacher forcing
                # print('decoder_input shape')
                # print(decoder_input.shape)
                #print('shape avant break')
                #print(decoder_input)
                if decoder_input.squeeze(0)[-1][0].item() == self.eos_token:  # Test le premier (batch de même size)
                    #print('break')
                    break
            #print('final shape')
            #print(torch.stack(decoder_outputs, dim=0).shape)
            #print('target_tensor shape')
            #print(target_tensor.shape)
            return torch.stack(decoder_outputs, dim=0), target_tensor, encoder_hidden
        else:
            # Without teacher forcing: use its own predictions as the next input
            breakable = [0]*batch_size
            for di in range(target_length+1): # +1 for SOS token
                # print(di)
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
                decoder_input = torch.cat([decoder_input, topi.permute(1, 0).detach()], dim=0)  # detach from history as input
                # print('decoder_input shape')
                # print(decoder_input.shape)
                for i in range(batch_size):
                    if decoder_input[-1][i].item() == self.eos_token:  # Test le premier (batch de même size)
                        breakable[i] = 1
                    if sum(breakable) == batch_size:
                        break
            return torch.stack(torch.unbind(torch.stack(decoder_outputs, dim=0), dim=0)[1:],
                                   dim=0), target_tensor, encoder_hidden

    def forward(self, source, need_embedding=True):

        batch_size = source[0].size(1)  # Batch size
        input_tensor = source[0]  # .squeeze(0)
        target_tensor = source[0]  # phrase d'entrée cible phrase de sortie .squeeze(0)

        encoder_hidden, encoder_outputs = self.encode(input_tensor, batch_size, need_embedding)
        print('batch size')
        print(batch_size)
        print('encoder output')
        print(encoder_outputs.shape)
        print('encoder hidden')
        print(encoder_hidden.shape)
        output, target, encoder_hidden = self.decode(encoder_hidden, encoder_outputs, batch_size, target_tensor)

        return output, target, encoder_hidden
