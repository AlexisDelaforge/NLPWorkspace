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
