import torch

# All collate_fn functions, use to pad batch and pre-process batch.

# My function


def classic_collate_fn(batch):
    # global device
    max_size = batch[0].size()
    max_length = max([s.size(0) for s in batch])
    embedding_dim = 300  # get from dataset class !!! /!\
    padding_tensor = torch.tensor([0] * embedding_dim, dtype=torch.float)  # .to(device) # go to global
    for sentence in range(len(batch)):
        if max_length != batch[sentence].shape[0]:
            padded_tensor = torch.stack([padding_tensor] * (max_length - batch[sentence].shape[0]), dim=0)  # .to(device)
            batch[sentence] = torch.cat((batch[sentence], padded_tensor), dim=0)
    return batch
