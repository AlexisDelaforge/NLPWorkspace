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

print('position 0')
print(torch.cuda.memory_allocated(0))

parameters['embedder'] = embedder.W2VCustomEmbedding(**embedder_params).to(parameters['device'])

print('position 0bis')
print(torch.cuda.memory_allocated(0))  # 1033MiB

dataloader_params['dataset'] = dataset.YelpTweetDataset(
    # path='/home/alexis/Project/Data/NLP_Dataset/all_setences_en_processed.tsv',
    path='../Data/Yelp/',
    file_name='20review_binary',
    file_type='csv',
    device=parameters['device'],
    text_column='text',
    label_column='target')

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

encoder_params = dict(
    embedder=parameters['embedder'],
    dropout_p=0.1,
    device=parameters['device'],
    teacher_forcing_ratio=0.5,
    num_layers=2,
    bidirectional=False,
    encode_size=512,
    max_length=max(dataloader_params['dataset'].size)
)

model_params = dict(
    num_class=dataloader_params['dataset'].num_class
)

parameters['encoder_model'] = models.AttnAutoEncoderRNN(**encoder_params).to(parameters['device'])  #models.TransformerModel(**model_params).to(parameters['device'])
parameters['encoder_model'].load_state_dict(torch.load(str("./executions/FromGPU4_MediumFixed/models/Best_Model_Epoch_20.pt"), map_location=device))
# parameters['classifier_model'] = models.SentimentRNN(**classifier_params).to(parameters['device'])  #models.TransformerModel(**model_params).to(parameters['device'])
# parameters['model'] = models.EncoderClassifier(parameters['encoder_model'], parameters['classifier_model'], parameters['embedder'])
parameters['model'] = models.EncoderClassifierDecoder(parameters['encoder_model'], parameters['embedder'], model_params['num_class'], device)

# print('somme emb param')
# print(parameters['embedder'].weights)

cross_entropy = nn.CrossEntropyLoss()

name_execution = 'FromGPU4_EncoderUnique'

#with open("./executions/" + name_execution + "/model.pkl", 'rb') as f:
    #model = pkl.load(f)
model = parameters['model'].to(parameters['device'])  #models.TransformerModel(**model_params).to(parameters['device'])

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

print('yoyo')

#with open("./executions/" + name_execution + "/embedder.pkl", 'rb') as f:
    #embedder = pkl.load(f)
for f in glob.glob("./executions/" + str(name_execution) + "/models/Model_Epoch_5.pt"):
    print('model import : '+str(f))
    model.load_state_dict(torch.load(str(f), map_location=device))
# model = torch.load(str("executions/FromGPU4_Short/models/Best_Model_Epoch_18.pt"))
model.eval()
embedder = model.embedder
embedder.to(device)


# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

print('somme emb param')
print(sum(sum(embedder.weights)))

#print(len(embedder.index2word))
embedder.index2word = {v: k for k, v in embedder.word2index.items()}
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print (name, param.data)
sentence_to_test = [
    # ['the','virus','is','still','out','there','.'],
    # ['montpellier','is','a','city','in','france','.'],
    # ['fuck','uno','new','york','diabetes','labrador','.'],
    # ['i','will','never','let','you','down','.'],
    # ['indeed', ',', 'i', 'knew', 'it', 'well', '.'],
    # ['dolphins','can','not','hide','from','us','.'],
    # ['i','am','a','phd','student','in','neural','network','.'],
    # ['is', 'anyone', 'interested', 'in', 'medical', 'deep', 'networking', 'with', 'nlp', '.'],
    # ['i', 'am', 'looking', 'for', 'a', 'data', 'analytics', 'position', '.'],
    # ['academic', 'researchers', 'need', 'to', 'worry', 'about', 'deep', 'learning', 'models', '!'],
    # ['<unk>']

    ['it','is','wonderful','.'],
    ['i','loved','it','.'],
    ['best','restaurant','ever','.'],
    ['do','not','go','.'],
    ['it','was','disgusting','.'],
    ['better','eat','shit','.']
]
# target = training_functions.word_to_idx(sentence_to_test, embedder.word2index).to(device)
# input = embedder(target)
#print(target.shape)
#print(input.shape)
#for sentence in sentence_to_test:

target = training_functions.word_to_idx(sentence_to_test, embedder.word2index).to(device)
target = target.transpose(1,0).to(device)
print(target.transpose(1,0).shape)
output, encoder_hidden, value_out, class_out = model(tuple([target, target]))
print('cross_entropy')
print(output.shape)
print(target.shape)
loss = 0
for di in range(len(output)):
    # print(str(output[di].shape)+" "+str(target[di].shape))
    loss += cross_entropy(output[di], target[di])  # voir pourquoi unsqueeze
loss = loss/len(output)
F1_loss = functions.F1_Loss_Sentences(len(parameters['embedder'].word2index), device)
F1_l = F1_loss(output.clone(), target.clone())
print('F1_loss')
print(F1_l[0])
print(F1_l[1])
print(F1_l[2])
print('loss')
print(loss)
print(target.shape)
print(output.shape)
#print(target.shape)
sentences, values = training_functions.tensor_to_sentences(output, embedder.index2word)
print(sentence_to_test)
print(sentences)
print('target')
print()
print(class_out)
print(values)