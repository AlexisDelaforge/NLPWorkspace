from torch.utils.data import Dataset
import pandas as pd
from functions.utils import Vocabulary
from tokenizer import tokenizer as tk


# My Code


class AllSentencesDataset(Dataset):  # A retravailler
    def __init__(self, path, text_column=1, id_column=None, name=None, sos='<sos>', eos='<eos>',
                 pad='<pad>', unk='<unk>', tok_type='spacy'):
        self.name = name
        self.path = path
        if id_column is None:
            self.data = pd.read_csv(path, usecols=[text_column], sep='\t')
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
            text = [str(i.text) for i in list(self.tokenizer(sample['text']))]
            target = sample['text']
            id = sample['id']
        else:
            sample = self.data.loc[index]
            text = [str(i.text) for i in list(self.tokenizer(sample['text']))]
            target = text

        #sample['features'] = [self.sos] + list(map(str, list(self.tokenizer(sample['features'])))) + [self.eos]
        #print(list(map(str, self.tokenizer(sample['features']))))
        return text, target

    def __len__(self):
        return int(len(self.data))

    def set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary

    def set_embedder(self, embedder):
        self.embedder = embedder
        self.vocabulary = embedder.vocabulary

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
