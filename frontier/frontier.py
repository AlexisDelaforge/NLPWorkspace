import torch
import torch.functional as F
from torch.nn.modules.loss import _Loss
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer, required
from sklearn.metrics import mean_absolute_error


def derive_data(model, batch):
    model.parameters.requires_grad = False # Parametres du model fixe
    batch[0].requires_grad = True # Parametres des données qui vont changer
    criterion = ClassesDistance()
    lr = 0.02
    # for word in
    optimizer = SGD_to_Frontier(batch[1], lr)
    diff = float('inf')
    epsilon = 0.001
    while diff > epsilon:
        output, encoder_hidden, class_out = model(batch)
        diff, classification = criterion(class_out, batch[1])
        diff.backward()
        optimizer.step(classification)

class ClassesDistance(Module):
    # https: // pytorch.org / docs / stable / _modules / torch / nn / modules / loss.html  # L1Loss

    __constants__ = ['reduction']

    def __init__(self, distance_function=mean_absolute_error):
        self.distance_function = distance_function

    def forward(self, output, target):  # Gère seulement le cas de deux classes
        target_tensor = torch.zeros(output.size(0))
        target_tensor[target] = 1
        distance = self.distance_function(output[0], output[1])
        if target == torch.max(output)[1]:  # si la classe est bien prédite
            return distance, True  # a voir ou est le moins
            # - distance car on veut aller dans le sens contraire de le bonne prédiction dès que la prédiction est bonne
        else:  # sinon
            return distance, False


class SGD_to_Frontier(Optimizer):

    # https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, good_prediction=True, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # good_prediction, si la phrase à dériver est bien classée par le réseau
        direction=1
        if good_prediction:
            direction=-1
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0: # Partie à faire
                    # d_p = d_p.add(p, alpha=weight_decay)
                    d_p = d_p.add(p, alpha=weight_decay*direction)
                if momentum != 0: # Partie à faire
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        # d_p = d_p.add(buf, alpha=momentum)
                        d_p = d_p.add(buf, alpha=momentum*direction)
                    else:
                        d_p = buf

                # p.add_(d_p, alpha=-group['lr'])
                p.add_(d_p, alpha=-group['lr']*direction)

        return loss
