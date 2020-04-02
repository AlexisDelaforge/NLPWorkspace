# List all tokenizer that I will use [to do]
# Peut-être à faire en classses (à voir)

import spacy
from spacy.tokenizer import Tokenizer

def tokenizer(type):
    if type == 'spacy':
        nlp = spacy.load("en_core_web_sm")
        tokenizer = nlp.tokenizer
        sos = [{'ORTH': "<sos>"}]
        eos = [{'ORTH': "<eos>"}]
        pad = [{'ORTH': "<pad>"}]
        unk = [{'ORTH': "<unk>"}]
        tokenizer.add_special_case("<sos>", sos)
        tokenizer.add_special_case("<eos>", eos)
        tokenizer.add_special_case("<pad>", pad)
        tokenizer.add_special_case("<unk>", unk)
        return tokenizer
    else:
        return None
