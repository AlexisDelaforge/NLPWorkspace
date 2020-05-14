import torch

def sentences_idx_to_word(sentences_idx, vocab):
    idx_word = {v: k for k, v in vocab.items()}
    idx_word[0] = '<unk>'
    sentences = []
    if len(sentences_idx.shape) != 1:
        for sentence_idx in sentences_idx :
            sentence = [idx_word[int(idx)] for idx in sentence_idx]
            sentences.append(sentence)
    else:
        sentences = [idx_word[int(idx)] for idx in sentences_idx]
    return sentences

def word_to_idx(sentences, vocab):
    longest_sentences = max(list(map(len, sentences)))
    #print(longest_sentences)
    sentences_idx = []
    for sentence in sentences :
        sentence_idx = []
        for word in sentence:
            if word in vocab:
                sentence_idx.append(vocab[word])
            else:
                sentence_idx.append(vocab['<unk>'])
        #while len(sentence_idx) <= longest_sentences:
            #sentence_idx += [vocab['<pad>']]
        #sentence_idx += [vocab['<eos>']]
        sentences_idx.append(sentence_idx)
    return torch.tensor(sentences_idx) # batch x len sentences

def tensor_to_sentences_idx(tensors):
    sentences_idx = []
    for tensor in tensors.unbind(1):
        sentence_idx = torch.argmax(tensor, dim=1)
        print(sentence_idx.shape)
        sentences_idx.append(sentence_idx)
    return sentences_idx

def tensor_to_sentences(tensors, idx_to_word):
    sentences_idx = torch.argmax(tensors, dim=1, keepdim=True)
    # print(sentences_idx.shape)
    # print(sentences_idx.shape)
    # print(torch.topk(tensors, k=10, dim=1))
    sentences = []
    sentences_value = []
    #print('dans algo')
    #print(tensors.shape)
    for sentence_tensor in tensors.unbind(1):
        sentence = []
        sentence_value = []
        #print(sentence_tensor.shape)
        for word_tensor in sentence_tensor.unbind(0):
            #print(word_tensor.shape)
            topv, topi = word_tensor.topk(1)
            #print(topv)
            #print(topi)
            # print(word_tensor)
            sentence.append(idx_to_word[topi.item()])
            sentence_value.append(topv.item())
        sentences.append(sentence)
        sentences_value.append(sentence_value)
    return sentences, sentences_value
