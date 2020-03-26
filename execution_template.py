import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import functions
from models import TransformerModel

parameters = dict({
    'execution_name': "name",
    'epochs': "number_of_epochs",
    'lr': 'leraning_rate'
})

functions.save_execution_file(parameters)

model = TransformerModel()

functions.add_to_execution_file(parameters, "no loss to compute")