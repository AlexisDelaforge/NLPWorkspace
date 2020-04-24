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
import time
import pickle as pkl
import glob
import os

# Set the device parameters
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
#print(device)

parameters = dict()
parameters['device'] = device
parameters['tmps_form_last_step'] = time.time()

# Should set all parameters of dataloader in this dictionary

dataloader_params = dict(
    dataset=None,  # Will change to take dataset
    batch_size=4,
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
    path='./data/model_embedding/fine_tune_W2V.model',
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
    vocab_size=len(parameters['embedder'].word2index),
    embed_size=parameters['embedder'].embedding_dim,
    sos_token=parameters['embedder'].word2index['<sos>'],
    eos_token=parameters['embedder'].word2index['<eos>'],
    dropout_p=0.1,
    device=parameters['device'],
    teacher_forcing_ratio=0.5,
    max_length=100
)


name_execution = 'FirstTestSeq2Seq'

#with open("./executions/" + name_execution + "/model.pkl", 'rb') as f:
    #model = pkl.load(f)
model = models.AttnAutoEncoderRNN(**model_params).to(parameters['device'])  #models.TransformerModel(**model_params).to(parameters['device'])
with open("./executions/" + name_execution + "/embedder.pkl", 'rb') as f:
    embedder = pkl.load(f)
for f in glob.glob("./executions/" + str(name_execution) + "/models/CPU_Best_Model_Epoch*.pt"):
    model.load_state_dict(torch.load(f, map_location=device))
model.eval().to(device)
embedder.to(device)
print(len(embedder.index2word))
embedder.index2word = {v: k for k, v in embedder.word2index.items()}
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print (name, param.data)
sentence_to_test = [['i', 'am', 'deeply', 'concerned', '.'],['your', 'arm', 'hurts', 'me', '.']]
target = training_functions.word_to_idx(sentence_to_test, embedder.word2index).to(device)
input = embedder(target)
print(target.shape)
print(input.shape)
output, target = model(tuple([input, target]))
print(output.shape)
print(training_functions.tensor_to_sentences(output, embedder.index2word))