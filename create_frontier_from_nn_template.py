import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import functions
import models
import embedder
import training_functions
from torch.utils import data
import glob
import dataset
from preprocessing import linear_interpolation_collate_fn
import time
import samplers
import frontier
import pandas as pd

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
    batch_size=2,
    shuffle=False,
    batch_sampler=samplers.OppositeSameSizeTwoSentenceBatchSampler,
    sampler=None,
    num_workers=0,
    collate_fn=linear_interpolation_collate_fn,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    divide_by=[1, 2, 5, 20],
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
    return_id=True,
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
    teacher_forcing_ratio=0,  # Non entrainement
    num_layers=2,
    bidirectional=False,
    encode_size=512,
    max_length=max(dataloader_params['dataset'].size)
)

# classifier_params = dict(
#     embedder=parameters['embedder'],
#     dropout=0.5,
#     layer_dropout=0.3,
#     device=parameters['device'], # a voir si je le laisse
#     n_layers=2,
#     bidirectional=False,
#     n_hidden=512,
#     n_out=2 #formule pour récupérer le nombre de classe du dataset
# )

model_params = dict(
    num_class=dataloader_params['dataset'].num_class
)

parameters['encoder_model'] = models.AttnAutoEncoderRNN(**encoder_params).to(parameters['device'])  #models.TransformerModel(**model_params).to(parameters['device'])
# parameters['encoder_model'].load_state_dict(torch.load(str("./executions/FromGPU4_MediumFixed/models/Best_Model_Epoch_20.pt"), map_location=device))
# parameters['classifier_model'] = models.SentimentRNN(**classifier_params).to(parameters['device'])  #models.TransformerModel(**model_params).to(parameters['device'])
# parameters['model'] = models.EncoderClassifier(parameters['encoder_model'], parameters['classifier_model'], parameters['embedder'])
parameters['model'] = models.EncoderClassifierDecoder(parameters['encoder_model'], parameters['embedder'], model_params['num_class'], device)

name_execution = 'FromGPU4_EncoderUnique' # A CHANGER

#with open("./executions/" + name_execution + "/model.pkl", 'rb') as f:
    #model = pkl.load(f)
parameters['model'] = parameters['model'].to(parameters['device'])  #models.TransformerModel(**model_params).to(parameters['device'])
parameters['encoder_model'] = parameters['model'].encoder
parameters['classifier_model'] = parameters['model'].classifier
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

#with open("./executions/" + name_execution + "/embedder.pkl", 'rb') as f:
    #embedder = pkl.load(f)
for f in glob.glob("./executions/" + str(name_execution) + "/models/Model_Epoch_5.pt"):
    print('model import : '+str(f))
    parameters['model'].load_state_dict(torch.load(str(f), map_location=device))
# model = torch.load(str("executions/FromGPU4_Short/models/Best_Model_Epoch_18.pt"))
parameters['model'].eval()
embedder = parameters['model'].embedder
embedder.to(device)

parameters['create_frontier_function'] = frontier.create_all_points_from_nn_frontier


# Should set all parameters of criterion in this dictionary

parameters['create_frontier_function'](parameters, './executions/FRONTIER/with_neigbourghs_V2.csv')