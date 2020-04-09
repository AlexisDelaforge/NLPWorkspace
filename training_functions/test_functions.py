import torch

def sentences_idx_to_word(sentences_idx, vocab):
    idx_word = {v: k for k, v in vocab.items()}
    idx_word[0] = '<unk>'
    sentences = []
    for sentence_idx in sentences_idx :
        sentence = [idx_word[int(idx)] for idx in sentence_idx]
        sentences.append(sentence)
    return sentences

def word_to_idx(sentences, vocab):
    longest_sentences = max(list(map(len,sentences)))
    #print(longest_sentences)
    sentences_idx = []
    for sentence in sentences :
        sentence_idx = [vocab['<sos>']]
        sentence_idx += [vocab[word] for word in sentence]
        while len(sentence_idx) <= longest_sentences:
            sentence_idx += [vocab['<pad>']]
        sentence_idx += [vocab['<eos>']]
        sentences_idx.append(sentence_idx)
    return torch.tensor(sentences_idx) # batch x len sentences

def tensor_to_sentences_idx(tensors):
    sentences_idx = []
    for tensor in tensors.unbind(1):
        sentence_idx = torch.argmax(tensor, dim=1)
        sentences_idx.append(sentence_idx)
    return sentences_idx

def tensor_to_sentences(tensors, idx_to_word):
    sentences_idx = torch.argmax(tensors, dim=1, keepdim=True)
    # print(sentences_idx.shape)
    sentences = []
    for sentence_tensor in sentences_idx.unbind(0):
        sentence = []
        for word_tensor in sentence_tensor.squeeze(0).unbind(0):
            # print(word_tensor)
            sentence.append(idx_to_word[word_tensor.item()])
        sentences.append(sentence)
    return sentences
