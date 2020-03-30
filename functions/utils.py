from tokenizer import tokenizer


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
    for i in listed_len:
        if i == (len(listed_len) - 1):
            listed_len[i] = len(dataset_length) - total_listed
        else:
            listed_len[i] = round(listed_len[i] * len(dataset_length))
            total_listed += listed_len[i]
    return listed_len


# My code

class Vocabulary:
    def __init__(self, name, eos='<eos>', sos='<sos>', unk='<unk>', pad='<pad>', tok_type='spacy', lower=True):
        self.tokenizer = tokenizer(tok_type)
        self.eos = eos
        self.sos = sos
        self.pad = pad
        self.unk = unk
        self.name = name
        self.lower = lower
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: self.sos, 1: self.eos, 2: self.pad, 3: self.unk}
        self.n_words = 4  # Count SOS and EOS

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
            self.n_words += 1
        else:
            self.word2count[word] += 1

# My code