from gensim.models import KeyedVectors, Word2Vec
import torch
from torch import nn
from functions.utils import Vocabulary


class W2VPTEmbedding(nn.Embedding):  # A completer sur le schema de W2VCustomEmbedding
    def __init__(self, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False,
                 _weight=None):
        # local
        self.path = '/home/alexis/Project/Data/WE_pretrained/word2vec/vectors/word2vec-negative300.model'
        # server
        # self.path = '/home/alexis/Project/Data/WE_pretrained/word2vec/vectors/word2vec-negative300.model'
        model = KeyedVectors.load_word2vec_format(self.path)
        self.weights = torch.FloatTensor(model.vectors)
        nn.Embedding.__init__(self, self.weights.shape[0], self.weights.shape[1], padding_idx=padding_idx,
                              max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                              sparse=sparse, _weight=torch.FloatTensor(model.vectors))


class W2VCustomEmbedding(nn.Embedding):
    def __init__(self, path, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False,
                 _weight=None):
        self.path = path
        #print('step 1 embed')
        self.model = Word2Vec.load(path)
        self.weights = torch.FloatTensor(self.model.wv.vectors)
        #print('step 2 embed')
        nn.Embedding.__init__(self, len(self.model.wv.vectors), self.model.wv.vector_size, padding_idx=padding_idx,
                              max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                              sparse=sparse, _weight=_weight)
        #print('step 3 embed')
        self.weights = torch.FloatTensor(self.model.wv.vectors)
        self.word2index = {key: self.model.wv.vocab[key].index for key in self.model.wv.vocab.keys()}
        self.index2word = {value: key for key, value in self.word2index.items()}
        self.word2count = {key: self.model.wv.vocab[key].count for key in self.model.wv.vocab.keys()}
        #print('step 4 embed')
        self.vocabulary = Vocabulary()
        #print('step 5 embed')
        self.vocabulary.import_vocabulary(self.word2index, self.word2count, 'W2VCustomEmbeddingVocabulary')
        #print('step 6 embed')

    def get_index(self, index):
        return self.word2index[index]
