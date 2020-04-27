import torch

# All collate_fn functions, use to pad batch and pre-process batch.

# My function


def classic_collate_fn(batch):
    # global device
    #print(type(batch))
    #print(len(batch))
    #print(type(batch[0]))
    #print(type(batch[1]))
    #print(type(batch[2]))
    new_batch = batch
    #print(type(new_batch[0]))
    #print(new_batch[0][0].shape)
    #print(new_batch[0][1].shape)
    #print(new_batch[0][2].shape)
    max_length = max([s[0].shape[0] for s in new_batch])
    #print(max_length)
    embedding_dim = new_batch[0][0].shape[1]  # get from dataset class !!! /!\
    padding_tensor = new_batch[0][2]  # .to(device) # go to global
    padded_id = new_batch[0][3]
    for sentence in range(len(new_batch)):
        if max_length != new_batch[sentence][0].shape[0]:
            padded_tensor = torch.stack([padding_tensor] * (max_length - new_batch[sentence][0].shape[0]), dim=0)  # .to(device)
            padded_sentence = torch.stack([padded_id] * (max_length - new_batch[sentence][1].shape[0]), dim=0)
            new_batch[sentence] = tuple([torch.cat([new_batch[sentence][0], padded_tensor], dim=0), torch.cat([new_batch[sentence][1], padded_sentence], dim=0)])
        else :
            new_batch[sentence] = tuple([new_batch[sentence][0], new_batch[sentence][1]])
    #print(new_batch[0][0].shape)
    #print(new_batch[0][1].shape)
    #print(new_batch[1][0].shape)
    #print(new_batch[1][1].shape)
    #print(new_batch[2][0].shape)
    #print(new_batch[2][1].shape)
    #print(type(new_batch))
    #print(len(new_batch[0]))
    #print(len(new_batch[2]))
    #print(len(new_batch[1]))

    return tuple([torch.stack([s[0] for s in new_batch]).permute(1, 0, 2), torch.stack([s[1] for s in new_batch]).permute(1, 0)])
    # tuple ( N_sentences, N_words, N_embedding / N_sentences, N_words )


def token_collate_fn(batch):
    # global device
    # print(type(batch))
    # print(len(batch))
    # print(type(batch[0]))
    # print(type(batch[1]))
    # print(type(batch[2]))
    new_batch = batch
    #print(type(new_batch[0]))
    #print(new_batch[0][0].shape)
    #print(new_batch[0][1].shape)
    #print(new_batch[0][2].shape)
    max_length = max([s[0].shape[0] for s in new_batch])
    #print(max_length)
    padded_id = new_batch[0][3]
    for sentence in range(len(new_batch)):
        if max_length != new_batch[sentence][0].shape[0]:
            padded_tensor = torch.stack([padded_id] * (max_length - new_batch[sentence][0].shape[0]), dim=0)  # .to(device)
            padded_sentence = torch.stack([padded_id] * (max_length - new_batch[sentence][1].shape[0]), dim=0)
            new_batch[sentence] = tuple([torch.cat([new_batch[sentence][0], padded_tensor], dim=0), torch.cat([new_batch[sentence][1], padded_sentence], dim=0)])
        else:
            new_batch[sentence] = tuple([new_batch[sentence][0], new_batch[sentence][1]])
    #print(new_batch[0][0].shape)
    #print(new_batch[0][1].shape)
    #print(new_batch[1][0].shape)
    #print(new_batch[1][1].shape)
    #print(new_batch[2][0].shape)
    #print(new_batch[2][1].shape)
    #print(type(new_batch))
    #print(len(new_batch[0]))
    #print(len(new_batch[2]))
    #print(len(new_batch[1]))

    return tuple([torch.stack([s[0] for s in new_batch]).permute(1, 0), torch.stack([s[1] for s in new_batch]).permute(1, 0)])
    # tuple ( N_sentences, N_words, N_embedding / N_sentences, N_words )

def token_collate_fn_same_size(batch):

    new_batch = batch
    for sentence in range(len(new_batch)):
        new_batch[sentence] = tuple([new_batch[sentence][0], new_batch[sentence][1]])
    return tuple([torch.stack([s[0] for s in new_batch]).permute(1, 0), torch.stack([s[1] for s in new_batch]).permute(1, 0)])
    # tuple ( N_sentences, N_words, N_embedding / N_sentences, N_words )