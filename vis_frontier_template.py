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
from preprocessing import token_collate_fn_same_size_target
import time
import samplers

# Set the device parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
print('Device in use : '+str(device))

# Create the parameters dict, will be fill after

parameters = dict()
parameters['device'] = device
parameters['tmps_form_last_step'] = time.time()

# Should set all parameters of dataloader in this dictionary

dataloader_params = dict( # A REVOIR POUR LES DONNEES TWEETS
    dataset=None,  # Will change to take dataset
    batch_size=60,
    shuffle=False,
    batch_sampler=samplers.GroupedBatchSampler,
    sampler=None,
    num_workers=0,
    collate_fn=token_collate_fn_same_size_target,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    divide_by=[1, 2, 5, 30],
    divide_at=[0, 20, 30, 50]
)

# Should set all parameters of criterion in this dictionary

embedder_params = dict(
    path='./data/model_embedding/fine_tune_W2V.model',
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    _weight=None
)

parameters['embedder'] = embedder.W2VCustomEmbedding(**embedder_params).to(parameters['device'])

dataloader_params['dataset'] = dataset.YelpTweetDataset(
    # path='/home/alexis/Project/Data/NLP_Dataset/all_setences_en_processed.tsv',
    path='../Data/Yelp/',
    file_name='20review_binary',
    file_type='csv',
    device=parameters['device'],
    text_column='text',
    label_column='target')

# Set True or False for padable

dataloader_params['dataset'].set_embedder(parameters)

parameters['pad_token'] = parameters['embedder'].word2index['<pad>']

# Should set all parameters of model in this dictionary

'''model_params = dict(
    ntoken=len(parameters['embedder'].word2index),  # len(TEXT.vocab.stoi), # the size of vocabulary
    ninp=parameters['embedder'].embedding_dim,  # embedding dimension
    nhid=512,  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers=6,  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder 10-16
    nhead=10,  # the number of heads in the multi_head_attention models
    dropout=0.1,
    device=parameters['device']
)'''

print('Longer sentence in data : '+str(max(dataloader_params['dataset'].size)))

encoder_params = dict(
    embedder=parameters['embedder'],
    dropout_p=0.1,
    device=parameters['device'],
    teacher_forcing_ratio=0.5,
    num_layers=1,
    bidirectional=False,
    encode_size=512,
    max_length=max(dataloader_params['dataset'].size)
)

classifier_params = dict(
    embedder=parameters['embedder'],
    dropout=0.5,
    layer_dropout=0.3,
    device=parameters['device'], # a voir si je le laisse
    n_layers=2,
    bidirectional=False,
    n_hidden=512,
    n_out=2 #formule pour récupérer le nombre de classe du dataset
)

parameters['encoder_model'] = models.AttnAutoEncoderRNN(**encoder_params).to(parameters['device'])  #models.TransformerModel(**model_params).to(parameters['device'])
parameters['encoder_model'].load_state_dict(torch.load(str("./executions/FromGPU4_MediumFixed/models/Best_Model_Epoch_20.pt"), map_location=device))
parameters['classifier_model'] = models.SentimentRNN(**classifier_params).to(parameters['device'])  #models.TransformerModel(**model_params).to(parameters['device'])
parameters['model'] = models.EncoderClassifier(parameters['encoder_model'], parameters['classifier_model'], parameters['embedder'])

# Should set all parameters of criterion in this dictionary

criterion_params = dict(
    ignore_index=parameters['pad_token'],  # ignore pad_token because it's not relevant
    weight=torch.tensor(functions.weight_cross(dataloader_params['dataset'].embedder.word2count), device=parameters['device'])
)

print(criterion_params['weight'][0])

parameters['encoder_criterion'] = nn.CrossEntropyLoss(**criterion_params).to(parameters['device'])
parameters['classifier_criterion'] = nn.CrossEntropyLoss().to(parameters['device'])
parameters['criterion'] = [parameters['encoder_criterion'], parameters['classifier_criterion']]

print('position 3')
print(torch.cuda.get_device_properties(0).total_memory)

# Should set all parameters of optimizer in this dictionary

parameters['encoder_lr'] = 1  # Always
parameters['classifier_lr'] = 1  # Always

encoder_optimizer_params = dict(
    params=parameters['encoder_model'].parameters(),
    lr=parameters['encoder_lr'],  # will change to take parameters['lr']
    momentum=0,
    dampening=0,
    weight_decay=0,
    nesterov=False
)

classifier_optimizer_params = dict(
    params=parameters['classifier_model'].parameters(),
    lr=parameters['classifier_lr'],  # will change to take parameters['lr']
    momentum=0,
    dampening=0,
    weight_decay=0,
    nesterov=False
)

# for name, param in parameters['model'].named_parameters():
#     if param.requires_grad:
#         print (name, param.data)

parameters['encoder_optimizer'] = torch.optim.SGD(**encoder_optimizer_params)
parameters['classifier_optimizer'] = torch.optim.SGD(**classifier_optimizer_params)
parameters['optimizer'] = dict(
    encoder=parameters['encoder_optimizer'],
    classifier=parameters['classifier_optimizer']
)

# Should set all parameters of scheduler in this dictionary

encoder_scheduler_params = dict(
    optimizer=parameters['encoder_optimizer'],  # will change to take parameters['optimizer']
    step_size=1,  # Each epoch do decay for 1, two epoch for 2 etc...
    gamma=0.9,  # Multiple lr by gamma value at each update
    last_epoch=-1
)

classifier_scheduler_params = dict(
    optimizer=parameters['classifier_optimizer'],  # will change to take parameters['optimizer']
    step_size=1,  # Each epoch do decay for 1, two epoch for 2 etc...
    gamma=0.9,  # Multiple lr by gamma value at each update
    last_epoch=-1
)

parameters['encoder_scheduler'] = torch.optim.lr_scheduler.StepLR(**encoder_scheduler_params)
parameters['encoder_scheduler_interval_batch'] = 1000000
parameters['classifier_scheduler'] = torch.optim.lr_scheduler.StepLR(**classifier_scheduler_params)
parameters['classifier_scheduler_interval_batch'] = 1000000
parameters['scheduler'] = dict(
    encoder=parameters['encoder_scheduler'],
    classifier=parameters['classifier_scheduler']
)
parameters['valid_interval_batch'] = 1000000
parameters['valid_interval_epoch'] = 1
parameters['l1_loss'] = False
if parameters['l1_loss']:
    print('l1_loss is True')

parameters['train_function'] = training_functions.encoder_classifier_train
parameters['collate_fn'] = token_collate_fn_same_size_target
parameters['execution_name'] = "VisFrontierFirstTests"  # Always
parameters['epochs'] = 100000  # Always
parameters['criterion_params'] = criterion_params
parameters['encoder_optimizer_params'] = encoder_optimizer_params
parameters['encoder_scheduler_params'] = encoder_scheduler_params
parameters['classifier_optimizer_params'] = classifier_optimizer_params
parameters['classifier_scheduler_params'] = classifier_scheduler_params
parameters['embedder_params'] = embedder_params
parameters['encoder_params'] = encoder_params
parameters['classifier_params'] = classifier_params
parameters['log_interval_batch'] = 200
# parameters['log_interval_batch'] = example de ligne que l'on veut retirer // Ligne à commenter
parameters['batch_size'] = dataloader_params['batch_size']  # Always
parameters['eval_batch_size'] = 10  # Always
parameters['split_sets'] = [.95, .025, .025]  # Use to set train, eval and test dataset size, should be egal to 1

functions.save_execution_file(parameters)

functions.add_to_execution_file(parameters, 'Code execute on ' + str(device))

functions.add_to_execution_file(parameters, 'Fin de définition des parametres en  ' + str(round((time.time()-parameters['tmps_form_last_step']), 2))+' secondes')
parameters['tmps_form_last_step'] = time.time()

print('position 4')
print(torch.cuda.get_device_properties(0).total_memory)  # 1768MiB

# print(dataloader_params['dataset'].vocabulary.word2index)
# print('aloha'+'\n')
# print(dataloader_params['dataset'].embedder.vocabulary.word2index)
# print(len(dataloader_params['dataset']))

train_data_loader, valid_data_loader, test_data_loader = functions.train_test_valid_dataloader(dataloader_params, parameters['split_sets'])

functions.add_to_execution_file(parameters, 'Début du chargement des data loader')
dataloader_time = time.time()

functions.add_to_execution_file(parameters, 'Fin de creation des dataloader en  ' + str(round((time.time()-dataloader_time), 2))+' secondes')


print('position 7')
print(torch.cuda.get_device_properties(0).total_memory)

functions.add_to_execution_file(parameters, 'Fin de creation des dataloader en  ' + str(round((time.time()-parameters['tmps_form_last_step']), 2))+' secondes')

functions.add_to_execution_file(parameters, 'Nombre de paramètres de l\'encoder : '+str(functions.count_parameters(parameters['encoder_model']))+' params')
functions.add_to_execution_file(parameters, 'Nombre de paramètres du classifier : '+str(functions.count_parameters(parameters['classifier_model']))+' params')

parameters['tmps_form_last_step'] = time.time()

#print(parameters['model'].device)

parameters['train_function'](parameters, train_data_loader, valid_data_loader)

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
