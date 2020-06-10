import torch
import torch.nn as nn
import torch.nn.functional as F
import training_functions
import random


# My code

class EncoderClassifier(nn.Module):
    def __init__(self, encoder, classifier, embedder):
        super(EncoderClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.embedder = embedder  # La déclaré a l'extérieur avec grad_false au préalable
        self.classifier.embedding.weight.requires_grad = False
        self.encoder.embedder.weight.requires_grad = False

    def forward(self, source, need_embedding=True):

        # print(source)
        output_tensor, target_tensor, encoder_hidden = self.encoder(source, need_embedding)
        value_out, class_out, class_hiddens = self.classifier(source, need_embedding)


        # print('output_tensor')
        # print(output_tensor.shape)
        # print('target_tensor')
        # print(target_tensor.shape)
        # print('encoder_hidden')
        # print(encoder_hidden.shape)
        # print('class_out')
        # print(class_out.shape)
        # print('class_hiddens[0][0]')
        # print(class_hiddens[0][0].shape)
        # print('class_hiddens[0][1]')
        # print(class_hiddens[0][1].shape)
        # print('class_hiddens[1]')
        # print(class_hiddens[1].shape)


        # X = torch.cat((x1, x2), dim=0)
        # X = self.emb(X)
        # x1, x2 = X[:B, ...], X[B:, ...]
        return output_tensor, encoder_hidden, value_out, class_out

class EncoderClassifierDecoder(nn.Module):
    def __init__(self, encoder, embedder, num_classes, device):
        super(EncoderClassifierDecoder, self).__init__()
        self.encoder = encoder
        self.embedder = embedder  # La déclaré a l'extérieur avec grad_false au préalable
        self.device = device
        self.classifier = torch.nn.Linear(self.encoder.encode_size*self.encoder.num_layers, num_classes, bias=True).to(self.device)
        # self.sig_out = nn.Sigmoid().to(self.device)
        self.sig_out = nn.Softmax(1).to(self.device)
        self.encoder.embedder.weight.requires_grad = False

    def forward(self, source, need_embedding=True):

        # print(source)
        output_tensor, target_tensor, encoder_hidden = self.encoder(source, need_embedding)
        # print(encoder_hidden.shape)
        encoder_hidden = torch.cat([tenseur.reshape(self.encoder.encode_size*self.encoder.num_layers).unsqueeze(1) for tenseur in encoder_hidden.unbind(1)], dim=1) #tenseur, dim=0)
        encoder_hidden = encoder_hidden.transpose(1, 0)
        # print(encoder_hidden)
        # print(encoder_hidden.shape)
        value_out = self.classifier(encoder_hidden)
        class_out = self.sig_out(value_out)

        # print('output_tensor')
        # print(output_tensor.shape)
        # print('target_tensor')
        # print(target_tensor.shape)
        # print('encoder_hidden')
        # print(encoder_hidden.shape)
        # print('class_out')
        # print(class_out.shape)
        # print('class_hiddens[0][0]')
        # print(class_hiddens[0][0].shape)
        # print('class_hiddens[0][1]')
        # print(class_hiddens[0][1].shape)
        # print('class_hiddens[1]')
        # print(class_hiddens[1].shape)


        # X = torch.cat((x1, x2), dim=0)
        # X = self.emb(X)
        # x1, x2 = X[:B, ...], X[B:, ...]
        return output_tensor, encoder_hidden, value_out, class_out
