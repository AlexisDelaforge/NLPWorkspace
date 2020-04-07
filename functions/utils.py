from tokenizer import tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

# Own modules to handle dictionary

# My code


def dict_less(dictionary, argument_to_forget):
    new_dictionary = dict(dictionary)
    for argument in argument_to_forget:
        try:
            new_dictionary.pop(argument)
        except KeyError:
            print('KeyError in argument to forget in dict_less function')
    return new_dictionary


# My code


def dict_keep(dictionary, argument_to_keep):
    new_dictionary = dict()
    for argument in argument_to_keep:
        try:
            new_dictionary[argument] = dictionary[argument]
        except KeyError:
            print('KeyError in argument to keep in dict_keep function')
    return new_dictionary


# My code


def dict_change(dictionary, argument_to_update):
    new_dictionary = dict(dictionary)
    for key, value in argument_to_update.items():
        try:
            new_dictionary[key] = value
        except KeyError:
            print('KeyError in argument to update in dict_change function')
    return new_dictionary


# My code

def split_values(dataset_length, listed_len):
    total_listed = 0
    k = 0
    for i in listed_len:
        if i == (len(listed_len) - 1):
            listed_len[k] = dataset_length - total_listed
        else:
            listed_len[k] = round(listed_len[k] * dataset_length)
            total_listed += listed_len[k]
        k += 1
    if sum(listed_len) < dataset_length:
        listed_len[0] += 1
    elif sum(listed_len) > dataset_length:
        listed_len[0] -= 1
    return listed_len


# My code

class Vocabulary:
    def __init__(self, name=None, eos='<eos>', sos='<sos>', unk='<unk>', pad='<pad>', tok_type='spacy', lower=True):
        self.tokenizer = tokenizer(tok_type)
        self.eos = eos
        self.sos = sos
        self.pad = pad
        self.unk = unk
        self.name = name
        self.lower = lower
        self.word2index = {self.sos: 0, self.eos: 1, self.pad: 2, self.unk: 3}
        self.word2count = {}
        self.index2word = {0: self.sos, 1: self.eos, 2: self.pad, 3: self.unk}
        self.n_words = 4  # Count SOS and EOS

    def build_vocab(self, sentences, min_count = 1):
        for sentence in sentences:
            self.add_sentence(sentence)

    def add_sentence(self, sentence):
        for word in list(tokenizer(sentence)):
            self.add_word(word)

    def add_word(self, word):
        if self.lower:
            word = str.lower(word)  # always use lower form of words
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.word2index[word] = self.n_words
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def import_vocabulary(self, word2index, word2count, name=None):
        self.name = name
        self.word2index = word2index
        self.index2word = {value: key for key, value in self.word2index.items()}
        self.n_words = len(self.word2index)
        self.word2count = word2count

# Not my code : https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354

class F1_Loss_Sentences(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, num_classes, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        device = 'cuda'
        # assert y_pred.ndim == 2
        # assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.num_classes).permute(0, 2, 1).to(torch.torch.float16)
        # print(y_pred.shape)
        topk, indices = torch.topk(y_pred, 1)
        y_pred = torch.zeros(y_pred.shape).to(device).scatter(1, indices, topk) / y_pred

        # y_pred = F.softmax(y_pred, dim=1)
        # print(y_pred.shape)
        tp = (y_true * y_pred).sum(dim=0).to(torch.torch.float16)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.torch.float16)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.torch.float16)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.torch.float16)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return (1 - f1.mean()), precision.mean(), recall.mean()


# f1_loss = F1_Loss().cuda()