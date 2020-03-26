import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import functions
from models import TransformerModel

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
    lr=None, # will change to take parameters['lr']
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

parameters = dict(
    execution_name="TextExecution",
    epochs="number_of_epochs",
    lr=5,
    criterion='criterion',
    criterion_params=criterion_params,
    optimizer_params=optimizer_params,
    scheduler_params=scheduler_params,
    model_params=model_params,
    batch_size = 20,
    eval_batch_size = 10,
)

parameters['model'] = TransformerModel(**parameters['model_params'])
parameters['criterion'] = nn.CrossEntropyLoss()
parameters['optimizer'] = torch.optim.SGD(parameters['model'].parameters(), **functions.dict_change(optimizer_params, {'lr':parameters['lr']}))
parameters['scheduler'] = torch.optim.lr_scheduler.StepLR(parameters['optimizer'], step_size=1.0, **functions.dict_less(scheduler_params, ['optimizer','step_size']))

# Define the function to do for each batch
# The input form is :

# The output should have the form :
#

def one_train(batch):
    return batch

functions.save_execution_file(parameters)

#model = TransformerModel()

#model to import

#data = dataLoader=

functions.add_to_execution_file(parameters, "no loss to compute")