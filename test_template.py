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
from preprocessing import classic_collate_fn, token_collate_fn, token_collate_fn_same_size
import time
import pickle as pkl
import glob
import os
import samplers


# Set the device parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:0"
#print(device)

parameters = dict()
parameters['device'] = device
parameters['tmps_form_last_step'] = time.time()

# Should set all parameters of dataloader in this dictionary

dataloader_params = dict(
    dataset=None,  # Will change to take dataset
    batch_size=60,
    shuffle=False,
    batch_sampler=samplers.GroupedBatchSampler,
    sampler=None,
    num_workers=0,
    collate_fn=token_collate_fn_same_size,
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

print('position 0')
print(torch.cuda.memory_allocated(0))

parameters['embedder'] = embedder.W2VCustomEmbedding(**embedder_params).to(parameters['device'])

print('position 0bis')
print(torch.cuda.memory_allocated(0))  # 1033MiB

dataloader_params['dataset'] = dataset.AllSentencesDataset(
    # path='/home/alexis/Project/Data/NLP_Dataset/all_setences_en_processed.tsv',
    path='../Data/NLP_Dataset/',
    file_name='10all_setences_en',
    file_type='tsv',
    device=parameters['device'],
    text_column=1)

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

model_params = dict(
    embedder=parameters['embedder'],
    dropout_p=0.1,
    device=parameters['device'],
    teacher_forcing_ratio=0.5,
    bidirectional=False,
    max_length=10  # 99 pour le test
)

cross_entropy = nn.CrossEntropyLoss()

name_execution = 'FromGPU4'

#with open("./executions/" + name_execution + "/model.pkl", 'rb') as f:
    #model = pkl.load(f)
model = models.AttnAutoEncoderRNN(**model_params).to(parameters['device'])  #models.TransformerModel(**model_params).to(parameters['device'])
#with open("./executions/" + name_execution + "/embedder.pkl", 'rb') as f:
    #embedder = pkl.load(f)
embedder = parameters['embedder']
for f in glob.glob("./executions/" + str(name_execution) + "/models/Best_Model_Epoch38.pt"):
    print('model import : '+str(f))
    model.load_state_dict(torch.load(f, map_location=device))
model.eval().to(device)
embedder.to(device)
#print(len(embedder.index2word))
embedder.index2word = {v: k for k, v in embedder.word2index.items()}
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print (name, param.data)
sentence_to_test = [
    # ['the','virus','is','still','out','there','.'],
    # ['i','am','a','phd','student','in','neural','network','.'],
    # ['montpellier','is','a','city','in','france','.'],
    # ['fuck','uno','new','york','diabetes','labrador','.'],
    # ['i','will','never','let','you','down','.'],
    ['indeed', ',', 'i', 'knew', 'it', 'well', '.']
    # ['dolphins','can','not','hide','from','us','.'],
    # ['is', 'anyone', 'interested', 'in', 'medical', 'deep', 'networking', 'with', 'nlp', '.'],
    # ['i', 'am', 'looking', 'for', 'a', 'data', 'analytics', 'position', '.'],
    # ['academic', 'researchers', 'need', 'to', 'worry', 'about', 'deep', 'learning', 'models', '!'],
    # ['<unk>']
]
# target = training_functions.word_to_idx(sentence_to_test, embedder.word2index).to(device)
# input = embedder(target)
#print(target.shape)
#print(input.shape)
#for sentence in sentence_to_test:
target = training_functions.word_to_idx(sentence_to_test, embedder.word2index).to(device)
target = target.transpose(1,0)
print(target.transpose(1,0).shape)
output, target = model(tuple([target, target]))
print('cross_entropy')
print(output.shape)
print(target.shape)
loss = 0
for di in range(len(output)):
    # print(str(output[di].shape)+" "+str(target[di].shape))
    loss += cross_entropy(output[di], target[di])  # voir pourquoi unsqueeze
loss = loss/len(output)
print(loss)
#print(target.shape)
print(output.shape)
#print(target.shape)
sentences, values = training_functions.tensor_to_sentences(output, embedder.index2word)
print(sentence_to_test)
print(sentences)
print(values)