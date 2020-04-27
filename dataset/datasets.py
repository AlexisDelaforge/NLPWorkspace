from torch.utils.data import Dataset, Sampler
import pandas as pd
from functions.utils import Vocabulary
from tokenizer import tokenizer as tk
import torch
import os.path
from os import path as pathos
import pickle

# My Code

class AllSentencesDataset(Sampler):  # A retravailler
    def __init__(self, path, file_name, file_type,  device, text_column=1, id_column=None, name=None, sos='<sos>', eos='<eos>',
                 pad='<pad>', unk='<unk>', tok_type='spacy'):
        self.name = name
        self.path = path
        self.file = path+file_name+"."+file_type
        self.device = device
        print('step 1')
        if id_column is None:
            self.data = pd.read_csv(self.file, usecols=[text_column], sep='\t')  # , nrows=401)
            self.id_column = False
            self.data.columns = ['text']
        else:
            self.data = pd.read_csv(self.file, usecols=[text_column, id_column], sep='\t')
            self.id_column = True
            self.data.columns = ['text', 'id']
        print('step 2')
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.unk = unk
        self.tokenizer = tk(tok_type)
        self.vocabulary = None
        self.embedder = None
        print('step 3')
        if pathos.exists(self.path+file_name+"_batch_len.pkl"):
            self.size = pickle.load(open(self.path+file_name+"_batch_len.pkl", "rb"))
        else:
            self.size = [len(self.tokenizer(str(self.data['text'][i]).lower())) for i in range(len(self.data))]
            pickle.dump(self.size, open(self.path+file_name+"_batch_len.pkl", "wb"))
        # self.size = [len(self.tokenizer(str(self.data['text'][i]).lower())) for i in range(len(self.data))]
        print('step 4')

    def __getitem__(self, index):
        if self.id_column:  # A refaire quand le else est ok
            sample = self.data.loc[self.data['id'] == index]
            text = [self.vocabulary.word2index[str(i.text).lower()] for i in list(self.tokenizer(sample['text']))]
            target = sample['text']
            id = sample['id']
        else:
            sample = self.data.loc[index]
            text = [self.vocabulary.word2index[str(i.text).lower()] if str(
                i.text).lower() in self.vocabulary.word2index else self.unk for i in
                    list(self.tokenizer(sample['text']))]
            target = [self.vocabulary.word2index[str(i.text).lower()] if str(
                i.text).lower() in self.vocabulary.word2index else self.unk for i in
                      list(self.tokenizer(sample['text']))]

        # sample['features'] = [self.sos] + list(map(str, list(self.tokenizer(sample['features'])))) + [self.eos]
        # print(list(map(str, self.tokenizer(sample['features']))))
        # print(target)
        return torch.tensor(text).to(self.device), torch.tensor(target).to(self.device)
        #self.embedder(torch.tensor(self.pad).to(self.device)).to(self.device), torch.tensor(self.pad).to(self.device)

    def __len__(self):
        return int(len(self.data))

    def __iter__(self):
        print('\tcalling AllSentencesDataset:__iter__')
        return iter(range(len(self.data)))

    def set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary

    def set_embedder(self, parameters, padable=True):  # Voir pour le freeze des parameters of nn.Embedding
        print('position setembed 1')
        print(torch.cuda.memory_allocated(0))
        self.embedder = parameters['embedder']
        self.vocabulary = parameters['embedder'].vocabulary
        self.vocabulary.add_word('<sos>')
        self.sos = self.vocabulary.word2index['<sos>']
        self.vocabulary.add_word('<eos>')
        self.eos = self.vocabulary.word2index['<eos>']
        self.vocabulary.add_word('<unk>')
        self.unk = self.vocabulary.word2index['<unk>']
        print('position setembed 2')
        print(torch.cuda.memory_allocated(0))
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
        print('position setembed 3')
        print(torch.cuda.memory_allocated(0))
        self.vocabulary.index2word = {v: k for k, v in self.vocabulary.word2index.items()}
        # print(self.embedder.weight.shape)
        print('position setembed 4')
        print(torch.cuda.memory_allocated(0))
        parameters['embedder'].word2index = dict(self.vocabulary.word2index)
        print('position setembed 5')
        print(torch.cuda.memory_allocated(0))
        parameters['embedder'].index2word = dict(self.vocabulary.index2word)
        print('position setembed 6')
        print(torch.cuda.memory_allocated(0))
        self.embedder = parameters['embedder']

    def name(self, name=None):
        if name is None:
            return self.name
        else:
            self.name = name
            return self.name


class AllSentencesDataset_V1(Dataset):  # A retravailler
    def __init__(self, path, device, text_column=1, id_column=None, name=None, sos='<sos>', eos='<eos>',
                 pad='<pad>', unk='<unk>', tok_type='spacy'):
        self.name = name
        self.path = path
        self.device = device
        if id_column is None:
            self.data = pd.read_csv(path, usecols=[text_column], sep='\t') # , nrows=401)
            self.id_column = False
            self.data.columns = ['text']
        else:
            self.data = pd.read_csv(path, usecols=[text_column, id_column], sep='\t')
            self.id_column = True
            self.data.columns = ['text', 'id']
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.unk = unk
        self.tokenizer = tk(tok_type)
        self.vocabulary = None
        self.embedder = None

    def __getitem__(self, index):
        if self.id_column: # A refaire quand le else est ok
            sample = self.data.loc[self.data['id'] == index]
            text = [self.vocabulary.word2index[str(i.text)] for i in list(self.tokenizer(sample['text']))]
            target = sample['text']
            id = sample['id']
        else:
            sample = self.data.loc[index]
            text = [self.vocabulary.word2index[str(i.text)] if str(i.text) in self.vocabulary.word2index else self.unk for i in list(self.tokenizer(sample['text']))]
            target = [self.vocabulary.word2index[str(i.text)] if str(i.text) in self.vocabulary.word2index else self.unk for i in list(self.tokenizer(sample['text']))]

        #sample['features'] = [self.sos] + list(map(str, list(self.tokenizer(sample['features'])))) + [self.eos]
        #print(list(map(str, self.tokenizer(sample['features']))))
        #print(text)
        return self.embedder(torch.tensor(text)), torch.tensor(target).to(self.device), self.embedder(torch.tensor(self.pad)), torch.tensor(self.pad).to(self.device)

    def __len__(self):
        return int(len(self.data))

    def set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary

    def set_embedder(self, embedder): # Voir pour le freeze des parameters of nn.Embedding
        self.embedder = embedder
        self.vocabulary = embedder.vocabulary
        self.sos = self.vocabulary.word2index['<sos>']
        self.eos = self.vocabulary.word2index['<eos>']
        self.pad = self.vocabulary.word2index['<pad>']
        self.vocabulary.add_word('<unk>')
        self.unk = self.vocabulary.word2index['<unk>']
        self.vocabulary.index2word = {v: k for k, v in self.vocabulary.word2index.items()}
        #print(self.unk)
        #print(self.embedder.weights.shape)
        #print(torch.mean(self.embedder.weights, dim=0).unsqueeze(dim=0).shape)
        #print(torch.cat([self.embedder.weights, torch.mean(self.embedder.weights, dim=0).unsqueeze(dim=0)], dim=0).shape)
        self.embedder.weight = torch.nn.Parameter(torch.cat([self.embedder.weights,torch.mean(self.embedder.weights, dim=0).unsqueeze(dim=0)], dim=0))
        self.embedder.weights = torch.cat([self.embedder.weights,torch.mean(self.embedder.weights, dim=0).unsqueeze(dim=0)], dim=0)
        #print(self.embedder.weight.shape)

    def name(self, name=None):
        if name is None:
            return self.name
        else:
            self.name = name
            return self.name


# My Code, TextTargetDataset uses Pandas.read_csv to open a datafram with two columns Text and Target

class TextTargetDataset(Dataset):
    def __init__(self, path, text_column=0, target_column=1, id_column=None, sos='<sos>', eos='<eos>',
                 pad='<pad>', unk='<unk>', name=None):
        self.name = name
        self.path = path
        if id_column is None:
            self.data = pd.read_csv(path, usecols=[text_column, target_column])
            self.id_column = False
        else:
            self.data = pd.read_csv(path, usecols=[text_column, target_column, id_column])
            self.data.columns[id_column] = 'id'
            self.id_column = True
        self.data.columns[text_column] = 'text'
        self.data.columns[target_column] = 'target'
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.unk = unk
        self.vocabulary = Vocabulary(sos='<sos>', eos='<eos>', pad='<pad>', unk='<unk>', tok_type='spacy', lower=True)

    def __getitem__(self, index):
        if self.id_column:
            sample = self.data.loc[self.data['id'] == index]
            sample = {'features': sample['text'], 'target': sample['target'], 'id': sample['id']}
        else:
            sample = self.data.iloc(index)
            sample = {'features': sample['text'], 'target': sample['target']}
        sample['features'] = [self.sos] + list(self.vocabulary.tokenizer(sample['features'])) + [self.eos]
        return sample

    def __len__(self):
        return int(len(self.data))

    def name(self, name=None):
        if name is None:
            return self.name
        else:
            self.name = name
            return self.name

#    def create_split(self, split_list):

# class Batch():
# def __init__(self, path, text_column=0, target_column=1, header=0, sos='<sos>', eos='<eos>', pad='<pad>', unk='<unk>'):
