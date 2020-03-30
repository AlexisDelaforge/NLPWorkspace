from torch.utils import data
import pandas as pd
from functions.utils import Vocabulary
from tokenizer import tokenizer

# My Code


class AllSentencesDataset(data.Dataset): # A retravailler
    def __init__(self, path, embedder, name=None):
        self.path = path
        with open(self.path) as f:
            self.len =  sum(1 for line in f)
        self.embedder = embedder
        self.name = name

    def __getitem__(self, index):
        all_index = list(range(self.len))
        all_index.pop(index)
        all_index.pop(0)
        line = pd.read_csv(self.path, header=0, skiprows=all_index)
        line = ['<sos>']+list(tokenizer(line['Source'][0]))+['<eos>']
        return self.embedder.tensorize(line)

    def __len__(self):
        return self.len

    def name(self, name=None):
        if name is None :
            return self.name
        else :
            self.name = name
            return self.name

# My Code, TextTargetDataset uses Pandas.read_csv to open a datafram with two columns Text and Target

class TextTargetDataset(data.Dataset):
    def __init__(self, path, text_column = 0, target_column = 1, header=0, sos='<sos>', eos='<eos>', pad='<pad>', unk='<unk>'):
        self.path = path
        self.data = pd.read_csv(path, header=header)
        self.data.columns[text_column] = 'text'
        self.data.columns[target_column] = 'target'
        self.vocabulary = Vocabulary(sos='<sos>', eos='<eos>', pad='<pad>', unk='<unk>', tok_type='spacy',lower=True)
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.unk = unk

    def __getitem__(self, index):
        all_index = list(range(self.len))
        all_index.pop(index)
        all_index.pop(0)
        line = pd.read_csv(self.path, header=0, skiprows=all_index)
        line = ['<sos>']+list(tokenizer(line['Source'][0]))+['<eos>']
        return self.embedder.tensorize(line)

#    def create_split(self, split_list):

