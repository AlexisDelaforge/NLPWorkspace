# List all tokenizer that I will use [to do]
# Peut-être à faire en classses (à voir)

def tokenizer(type):
    if type == 'spacy':
        from spacy.tokenizer import Tokenizer
        from spacy.lang.en import English
        nlp = English()
        # Create a blank Tokenizer with just the English vocab
        return Tokenizer(nlp.vocab)
    else:
        return None
