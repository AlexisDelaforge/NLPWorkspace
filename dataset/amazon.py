from torch.utils.data import Dataset, Sampler
import pandas as pd
from functions.utils import Vocabulary
from tokenizer import tokenizer as tk
import torch
import os.path
from os import path as pathos
import pickle

class AmazonReview(Sampler):  # A retravailler REPLACE \n !!! /!\
    def __init__(self, path, file_name, file_type, device, return_id=False, text_column=7, label_column=[6, 8], id_column=12, name=None, sos='<sos>', eos='<eos>',
                 pad='<pad>', unk='<unk>', tok_type='spacy'):
        self.name = name
        self.path = path
        self.file = path+file_name+"."+file_type
        self.device = device
        if id_column is None:
            self.data = pd.read_csv(self.file, usecols=[text_column, label_column], sep=',') #, nrows=10000) #, nrows=4)  # , nrows=401)
            self.id_column = False
            if text_column < label_column:
                self.data.columns = ['text', 'target']
            else:
                self.data.columns = ['target', 'text']
        else:
            self.data = pd.read_csv(self.file, usecols=[text_column, label_column[0], label_column[1], id_column], sep=',')
            self.id_column = True
            self.data.columns = ['helpful', 'text', 'score', 'id']  # condition d'ordre à faire
            # print(self.data.iloc[0])
        #spécifique à AmazonReview
        self.num_class = 2
        # self.classes = dict()
        self.return_id = return_id
        # for i in range(self.num_class):
        self.classes = ['helpful', 'score']
        # print('step 2')
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.unk = unk
        self.tokenizer = tk(tok_type)
        self.vocabulary = None
        self.embedder = None
        # print('step 3')
        if pathos.exists(self.path+file_name+"_batch_len.pkl"):
            self.size = pickle.load(open(self.path+file_name+"_batch_len.pkl", "rb"))
            self.data['size'] = self.size # [:10000]
        else:
            self.size = [len(self.tokenizer(str(self.data['text'][i]).lower().replace("\n", ""))) for i in range(len(self.data))]
            pickle.dump(self.size, open(self.path+file_name+"_batch_len.pkl", "wb"))
            self.data['size'] = self.size
        # self.size = [len(self.tokenizer(str(self.data['text'][i]).lower())) for i in range(len(self.data))]
        # self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.id = self.data['id']
        self.size = self.data['size'] # [:4]
        self.helpful = self.data['helpful']
        self.score = self.data['score']
        # print(self.data.head())
        # print('step 4')
        print('sizes')
        print(len(self.size))
        # print(self.score)
        # print(len(self.helpful))


    def __getitem__(self, index):

        if self.id_column:  # A refaire quand le else est ok
            # print(index)
            sample = self.data.loc[self.data['id'] == index]
            # print(sample)
            # print(type(sample))
            if type(sample['text']) != 'str':
                # print('here')
                # var = sample.iloc[0]['text']
                # del sample
                # sample = {}
                # sample['text'] = var
                sample = sample.iloc[0]
            text = [self.vocabulary.word2index[str(i.text).lower().replace("\n", " ")] if str(
                i.text).lower().replace("\n", " ") in self.vocabulary.word2index else self.unk for i in
                    list(self.tokenizer(str(sample['text']).lower().replace("\n", " ")))]
            target = [sample[self.classes[0]], sample[self.classes[1]]]
            id = sample['id']
        else:
            sample = self.data.loc[self.data['id'] == index]
            # print(sample['text'])
            # print(sample['text'].lower().replace("\n", " "))
            text = [self.vocabulary.word2index[str(i.text).lower().replace("\n", " ")] if str(
                i.text).lower().replace("\n", " ") in self.vocabulary.word2index else self.unk for i in
                    list(self.tokenizer(str(sample['text']).lower().replace("\n", " ")))]
            target = [sample[self.classes[0]], sample[self.classes[1]]]
        # sample['features'] = [self.sos] + list(map(str, list(self.tokenizer(sample['features'])))) + [self.eos]
        # print(list(map(str, self.tokenizer(sample['features']))))
        # print(torch.tensor(text).to(self.device), torch.tensor(target).to(self.device))
        # print(len(torch.tensor(text)))
        if self.return_id:
            # print(torch.tensor(target).type(torch.LongTensor).to(self.device))
            return torch.tensor(text).to(self.device), torch.tensor(target).to(self.device), id
        else:
            # print(torch.tensor(target).to(self.device))
            return torch.tensor(text).to(self.device), torch.tensor(target).to(self.device)
        #self.embedder(torch.tensor(self.pad).to(self.device)).to(self.device), torch.tensor(self.pad).to(self.device)

    def __len__(self):
        return int(len(self.data))

    def __iter__(self):
        # print('\tcalling AllSentencesDataset:__iter__')
        return iter(range(len(self.data)))

    def set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary

    def shuffle(self):
        self.data = self.data.sample(frac=1).reset_index(drop=False)
        self.size = self.data['size']  # [:4]
        self.helpful = self.data['helpful']
        self.score = self.data['score']

    def set_embedder(self, parameters, padable=True):  # Voir pour le freeze des parameters of nn.Embedding
        # print('position setembed 1')
        # print(torch.cuda.memory_allocated(0))
        self.embedder = parameters['embedder']
        self.vocabulary = parameters['embedder'].vocabulary
        self.vocabulary.add_word('<sos>')
        self.sos = self.vocabulary.word2index['<sos>']
        self.vocabulary.add_word('<eos>')
        self.eos = self.vocabulary.word2index['<eos>']
        self.vocabulary.add_word('<unk>')
        self.unk = self.vocabulary.word2index['<unk>']
        # print('position setembed 2')
        # print(torch.cuda.memory_allocated(0))
        if padable:
            self.vocabulary.add_word('<pad>')
            self.pad = self.vocabulary.word2index['<pad>']
            # print(self.unk)
            # print(self.embedder.weights.shape)
            # print(torch.mean(self.embedder.weights, dim=0).unsqueeze(dim=0).shape)
            # print(torch.cat([self.embedder.weights, torch.mean(self.embedder.weights, dim=0).unsqueeze(dim=0)], dim=0).shape)
            parameters['embedder'].weight = torch.nn.Parameter(
                torch.cat([parameters['embedder'].weights, torch.mean(parameters['embedder'].weights, dim=0).unsqueeze(dim=0)], dim=0))
            parameters['embedder'].weights = torch.cat(
                [parameters['embedder'].weights, torch.mean(parameters['embedder'].weights, dim=0).unsqueeze(dim=0)], dim=0)
        # print('position setembed 3')
        # print(torch.cuda.memory_allocated(0))
        self.vocabulary.index2word = {v: k for k, v in self.vocabulary.word2index.items()}
        # print(self.embedder.weight.shape)
        # print('position setembed 4')
        # print(torch.cuda.memory_allocated(0))
        parameters['embedder'].word2index = dict(self.vocabulary.word2index)
        # print('position setembed 5')
        # print(torch.cuda.memory_allocated(0))
        parameters['embedder'].index2word = dict(self.vocabulary.index2word)
        # print('position setembed 6')
        # print(torch.cuda.memory_allocated(0))
        self.embedder = parameters['embedder']

    def name(self, name=None):
        if name is None:
            return self.name
        else:
            self.name = name
            return self.name

# test = AmazonReview('/home/alexis/Project/Data/AmazonReview/', 'data_processed', 'csv', 'cuda')
# print(test.data.id[0])
# print(test.data.helpful[0])
# print(test.data.score[0])
# print(test.data.text[0])