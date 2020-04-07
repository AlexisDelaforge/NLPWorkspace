import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import functions
import models
import embedder
import training_functions
from torch.utils import data
import dataset
from preprocessing import classic_collate_fn

# Set the device parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cuda:0"
print(device)

# Create the parameters dict, will be fill after

parameters = dict()
parameters['device'] = device

# Should set all parameters of dataloader in this dictionary

dataloader_params = dict(
    dataset=None,  # Will change to take dataset
    batch_size=26,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=classic_collate_fn,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None
)

# Should set all parameters of criterion in this dictionary

embedder_params = dict(
    path='./data/model_embedding/model.bin',
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    _weight=None
)

parameters['embedder'] = embedder.W2VCustomEmbedding(**embedder_params).to(parameters['device'])

dataloader_params['dataset'] = dataset.AllSentencesDataset(
    # path='/home/alexis/Project/Data/NLP_Dataset/all_setences_en_processed.tsv',
    path='../Data/NLP_Dataset/all_setences_en_processed.tsv',
    device=parameters['device'],
    text_column=1)

dataloader_params['dataset'].set_embedder(parameters['embedder'])
parameters['pad_token'] = parameters['embedder'].word2index['<pad>']

# Should set all parameters of model in this dictionary

model_params = dict(
    ntoken=len(parameters['embedder'].word2index),  # len(TEXT.vocab.stoi), # the size of vocabulary
    ninp=parameters['embedder'].embedding_dim,  # embedding dimension
    nhid=200,  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers=2,  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=2,  # the number of heads in the multi_head_attention models
    dropout=0.2,
    device=parameters['device']
)

parameters['model'] = models.TransformerModel(**model_params).to(parameters['device'])

# Should set all parameters of criterion in this dictionary

criterion_params = dict(
    ignore_index = parameters['pad_token']  # ignore pad_token because it's not relevant
)

parameters['criterion'] = nn.CrossEntropyLoss().to(parameters['device'])

# Should set all parameters of optimizer in this dictionary

parameters['lr'] = 0.8  # Always

optimizer_params = dict(
    params=parameters['model'].parameters(),
    lr=parameters['lr'],  # will change to take parameters['lr']
    momentum=0,
    dampening=0,
    weight_decay=0,
    nesterov=False
)

parameters['optimizer'] = torch.optim.SGD(**optimizer_params)

# Should set all parameters of scheduler in this dictionary

scheduler_params = dict(
    optimizer=parameters['optimizer'],  # will change to take parameters['optimizer']
    step_size=1.0,  # Each epoch do decay for 1, two epoch for 2 etc...
    gamma=0.8,  # Multiple lr by gamma value at each update
    last_epoch=-1
)

parameters['scheduler'] = torch.optim.lr_scheduler.StepLR(**scheduler_params)
parameters['scheduler_interval_batch'] = True
parameters['valid_interval_batch'] = 1000

parameters['execution_name'] = "PremierTestTransformerEncoder"  # Always
parameters['epochs'] = 10  # Always
parameters['criterion_params'] = criterion_params
parameters['optimizer_params'] = optimizer_params
parameters['scheduler_params'] = scheduler_params
parameters['embedder_params'] = embedder_params
parameters['model_params'] = model_params
parameters['log_interval_batch'] = 10
# parameters['log_interval_batch'] = example de ligne que l'on veut retirer // Ligne à commenter
parameters['batch_size'] = 20  # Always
parameters['eval_batch_size'] = 10  # Always
parameters['split_sets'] = [.98, .01, .01]  # Use to set train, eval and test dataset size, should be egal to 1

functions.save_execution_file(parameters)

functions.add_to_execution_file(parameters, 'Code execute on ' + str(device))

# print(dataloader_params['dataset'].vocabulary.word2index)
# print('aloha'+'\n')
# print(dataloader_params['dataset'].embedder.vocabulary.word2index)
# print(len(dataloader_params['dataset']))
train_set, valid_set, test_set = torch.utils.data.random_split(dataloader_params['dataset'],
                                                               functions.split_values(len(dataloader_params['dataset']),
                                                                                      parameters['split_sets']))

train_data_loader = data.DataLoader(**functions.dict_change(dataloader_params, {'dataset': train_set}))
valid_data_loader = data.DataLoader(**functions.dict_change(dataloader_params, {'dataset': valid_set}))
test_data_loader = data.DataLoader(**functions.dict_change(dataloader_params, {'dataset': test_set}))

training_functions.full_train(parameters, train_data_loader, valid_data_loader, None)

# Define the function to do for each batch
# The input form is :

print(test_data_loader.dataset.__len__())  # Voir pourquoi _= ça marche pas
print('\n')

for batch in test_data_loader:
    print(batch)


# print(parameters['embedder'].vocabulary())

# The output should have the form :
#

def one_train(batch):
    return batch


# model = TransformerModel()

# model to import

# data = dataLoader=

functions.add_to_execution_file(parameters, "no loss to compute")
