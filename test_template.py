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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cuda:0"
print(device)

# Create the parameters dict, will be fill after

parameters = dict()
parameters['device'] = device

name_execution = 'SecondTestTransformerEncoder'

with open("./executions/" + name_execution + "/model.pkl", 'rb') as f:
    model = pkl.load(f)
with open("./executions/" + name_execution + "/embedder.pkl", 'rb') as f:
    embedder = pkl.load(f)
for f in glob.glob("./executions/" + str(name_execution) + "/models/Best_Model_Epoch*.pt"):
    model.load_state_dict(torch.load(f, map_location=device))
model.eval().to(device)
embedder.to(device)
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