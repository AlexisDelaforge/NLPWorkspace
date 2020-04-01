import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import functions
from models import TransformerModel
from torch.utils import data
import dataset

# Set the device parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cuda:0"
print()

# Should set all parameters of model in this dictionary

model_params = dict(
    ntoken = 2342, #len(TEXT.vocab.stoi), # the size of vocabulary
    ninp = 200, # embedding dimension
    nhid = 200, # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2, # the number of heads in the multiheadattention models
    dropout = 0.2
)

# Should set all parameters of criterion in this dictionary

criterion_params = dict()

# Should set all parameters of optimizer in this dictionary

optimizer_params = dict(
    lr=None,  # will change to take parameters['lr']
    momentum=0,
    dampening=0,
    weight_decay=0,
    nesterov=False
)

# Should set all parameters of scheduler in this dictionary

scheduler_params = dict(
    optimizer=None, # will change to take parameters['optimizer']
    step_size=None, # Have more attention to this parameters
    gamma=0.1,
    last_epoch=-1
)

dataloader_params = dict(
    dataset=None, # Will change to take dataset
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None
)

parameters = dict(
    execution_name="TextExecution", # Always
    epochs=10, # Always
    lr=5, # Always
    criterion_params=criterion_params,
    optimizer_params=optimizer_params,
    scheduler_params=scheduler_params,
    model_params=model_params,
    batch_size = 20, # Always
    eval_batch_size = 10, # Always
    split_sets = [.90,.08,.02] # Use to set train, eval and test dataset size, should be egal to 1
)

functions.save_execution_file(parameters)

functions.add_to_execution_file(parameters, 'Code execute on '+str(device))

parameters['model'] = TransformerModel(**parameters['model_params'])
parameters['criterion'] = nn.CrossEntropyLoss()
parameters['optimizer'] = torch.optim.SGD(parameters['model'].parameters(), **functions.dict_change(optimizer_params, {'lr':parameters['lr']}))
parameters['scheduler'] = torch.optim.lr_scheduler.StepLR(parameters['optimizer'], step_size=1.0, **functions.dict_less(scheduler_params, ['optimizer','step_size']))

dataloader_params['dataset'] = dataset.AllSentencesDataset(path='/home/alexis/Project/Data/NLP_Dataset/all_setences_en_processed.tsv', text_column=0)

train_set, valid_set, test_set = torch.utils.data.random_split(dataloader_params['dataset'], functions.split_values(len(dataloader_params['dataset']), parameters['split_sets']))

train_data_loader = data.DataLoader(**functions.dict_change(dataloader_params, {'dataset': train_set}))
valid_data_loader = data.DataLoader(**functions.dict_change(dataloader_params, {'dataset': valid_set}))
test_data_loader = data.DataLoader(**functions.dict_change(dataloader_params, {'dataset': test_set}))

# Define the function to do for each batch
# The input form is :

print(test_data_loader.dataset.__len__()) # Voir pourquoi _= ça marche pas

# The output should have the form :
#

def one_train(batch):
    return batch

#model = TransformerModel()

#model to import

#data = dataLoader=

functions.add_to_execution_file(parameters, "no loss to compute")