import torch
import torch.functional as F
from torch.nn.modules.loss import _Loss
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer, required
from sklearn.metrics import mean_absolute_error
import time
import training_functions

def derive_data(model, batch):
    model.parameters.requires_grad = False  # Parametres du model fixe
    batch[0].requires_grad = True  # Parametres des données qui vont changer
    criterion = ClassesDistance()
    lr = 0.02
    # for word in
    optimizer = torch.optim.SGD(batch[0], lr)
    diff = float('inf')
    epsilon = 0.001
    while diff > epsilon:
        output, encoder_hidden, class_out = model(batch)
        diff, classification = criterion(class_out, batch[1])
        diff.backward()
        optimizer.step(classification)


class ClassesDistance(_Loss):
    # https: // pytorch.org / docs / stable / _modules / torch / nn / modules / loss.html  # L1Loss

    __constants__ = ['reduction']

    def __init__(self, distance_function=mean_absolute_error):
        super(ClassesDistance, self).__init__()
        self.distance_function = distance_function

    def forward(self, output, absolute):  # Gère seulement le cas de deux classes
        # target_tensor = torch.zeros(output.size(0))
        # target_tensor[target] = 1
        # print('gueuledelouput')
        # print(output)
        distance = output.transpose(1, 0)[0] - output.transpose(1, 0)[1]
        if absolute:
            distance = distance.abs()
        # if target == torch.max(output)[1]:  # si la classe est bien prédite
        #     return distance, True  # a voir ou est le moins
        #     # - distance car on veut aller dans le sens contraire de le bonne prédiction dès que la prédiction est bonne
        # else:  # sinon
        #     return distance, False
        # print(output[0].grad_fn)
        # print(output[1].grad_fn)
        return distance


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
        direction = 1
        if good_prediction:
            direction = -1
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
                if weight_decay != 0:  # Partie à faire
                    # d_p = d_p.add(p, alpha=weight_decay)
                    d_p = d_p.add(p, alpha=weight_decay * direction)
                if momentum != 0:  # Partie à faire
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        # d_p = d_p.add(buf, alpha=momentum)
                        d_p = d_p.add(buf, alpha=momentum * direction)
                    else:
                        d_p = buf

                # p.add_(d_p, alpha=-group['lr'])
                p.add_(d_p, alpha=-group['lr'] * direction)

        return loss


def encoder_classifier_frontier(parameters, train_data_loader, max_iter=1500000):
    start = time.time()
    # parameters['model'].parameters().requires_grad = False
    parameters['model'].train()
    for param in parameters['model'].parameters():  # Parametres du model fixe
        param.requires_grad = False
    criterion = ClassesDistance()
    batch_num = 0
    for batch in train_data_loader:  # parameters['batchs']
        batch_num += 1
        iter = 0
        # print(batch)
        # input = batch[0]
        input = parameters['embedder'](batch[0]).float()
        input.requires_grad = True  # Parametres des données qui vont changer
        print(input.requires_grad)
        print(input.shape)
        print(input)
        optimizer = torch.optim.SGD([input], parameters['lr'])
        diff = torch.tensor([float('inf')]*batch[0].size(0))
        epsilon = 0.1
        while sum(diff > epsilon).item() != 0 and iter < max_iter:
            iter += 1
            optimizer.zero_grad()
            # print(input.shape)
            output, encoder_hidden, value_out, class_out = parameters['model'](tuple([input, batch[1]]), False)
            diff = criterion(value_out)
            diff_soft = criterion(class_out)
            # print(diff.grad_fn)
            print(diff_soft.sum())
            print(diff.sum())
            diff.sum().backward()
            optimizer.step()
        else:
            if iter == max_iter:
                print("Le batch "+str(batch_num)+" est arrivé à la limite d'itération fixée à : "+str(max_iter))
            else:
                print("Le batch " + str(batch_num) + " a eu besoin de " + str(iter)+" itérations.")
            print("L'ouput est de la forme : ")
            print(parameters['model'](tuple([input, batch[1]]), False)[2].shape)
            print(parameters['model'](tuple([input, batch[1]]), False)[2])

    #     enc_target = batch[0]
    #     cls_target = batch[1]
    #     parameters['model'].train()
    #     # print(batch[0].shape)
    #     numb_sent += batch[0].size(1)
    #     # print('num sentence ajouté '+str(batch[0].size(1)))
    #     batch_num += 1
    #     parameters['batch_start_time'] = time.time()
    #     if type(parameters['optimizer']) is dict:
    #         for model, optimizer in parameters['optimizer'].items():
    #             optimizer.zero_grad()
    #     else:
    #         parameters['optimizer'].zero_grad()
    #     # print(batch_num)
    #     enc_loss = 0
    #     cls_loss = 0
    #     # print('batch structure')
    #     # print(batch[0].shape)
    #     # print(batch[1].shape)
    #     output, encoder_hidden, class_out = parameters['model'](batch)
    #     # functions.add_to_execution_file(parameters, 'La phrase et son output')
    #     # sentences, values = training_functions.tensor_to_sentences(output, parameters['embedder'].index2word)
    #     # functions.add_to_execution_file(parameters, str(target[0]))
    #     # functions.add_to_execution_file(parameters, str(sentences))
    #     # functions.add_to_execution_file(parameters, str(values))
    #     # functions.add_to_execution_file(parameters, 'Fin phrase et son output')
    #     for di in range(len(output)):
    #         # print(str(output[di].shape)+" "+str(enc_target[di].shape))
    #         enc_loss += parameters['encoder_criterion'](output[di], enc_target[di])  # voir pourquoi unsqueeze
    #         # batch[0][di] is the target
    #     enc_loss = enc_loss / len(output)
    #     # print(class_out.shape)
    #     # print(cls_target.shape)
    #     cls_loss += parameters['classifier_criterion'](class_out, cls_target)
    #     # print('/ len(target) before backward()') # /len(target) before backward()
    #
    #     enc_loss.backward()
    #     cls_loss.backward()
    #     if 'grad_norm' in parameters and parameters['grad_norm']:
    #         parameters.grad_norm(parameters['model'].parameters())
    #         # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    #         # functions.add_to_execution_file(parameters, 'Fin de grad_norm  ' + str(
    #         #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
    #         # parameters['tmps_form_last_step'] = time.time()
    #     if parameters['scheduler'] is not None:
    #         if type(parameters['scheduler']) is dict:
    #             for model, scheduler in parameters['scheduler'].items():  # créer dictionnaire plutot que liste
    #                 if model + '_scheduler_interval_batch' in parameters and batch_num % \
    #                         parameters[model + '_scheduler_interval_batch'] == 0 and batch_num != 0:
    #                     # print('step')
    #                     # print(parameters['scheduler'].get_last_lr())
    #                     scheduler.step()
    #         else:
    #             if 'scheduler_interval_batch' in parameters and batch_num % \
    #                     parameters[
    #                         'scheduler_interval_batch'] == 0 and batch_num != 0:
    #                 # print('step')
    #                 # print(parameters['scheduler'].get_last_lr())
    #                 parameters['scheduler'].step()
    #     if 'optimizer' in parameters:
    #         if type(parameters['optimizer']) is dict:
    #             for model, optimizer in parameters['optimizer'].items():
    #                 optimizer.step()
    #         else:
    #             parameters['optimizer'].step()
    #         # print(parameters['scheduler'].get_last_lr())
    #         # functions.add_to_execution_file(parameters, 'Fin de optimizer  ' + str(
    #         #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
    #         # parameters['tmps_form_last_step'] = time.time()
    #     '''for name, param in parameters['model'].named_parameters():
    #         if param.requires_grad:
    #             print(name, param.data)'''
    #
    #     # print(target)
    #     # print(loss)
    #     enc_total_loss += enc_loss.item()
    #     cls_total_loss += cls_loss.item()
    #     # print(batch_num)
    #     if 'log_interval_batch' in parameters and batch_num % parameters[
    #         'log_interval_batch'] == 0 and batch_num > 0:
    #         # print(batch_num % parameters['log_interval_batch'])
    #         # print(batch_num)
    #         if batch_num % parameters['log_interval_batch'] == 0 and batch_num > 0:
    #             # batch.num doit exister voir dataloader !!!
    #             prev_sent = numb_sent - prev_sent
    #             enc_cur_loss = enc_total_loss / parameters['log_interval_batch']
    #             cls_cur_loss = cls_total_loss / parameters['log_interval_batch']
    #             elapsed = time.time() - parameters['log_interval_time']
    #             if parameters['scheduler'] is not None:
    #                 if type(parameters['scheduler']) is dict:
    #                     for model, scheduler in parameters['scheduler'].items():
    #                         if model + '_scheduler_interval_batch' in parameters and batch_num % \
    #                                 parameters[model + '_scheduler_interval_batch'] == 0 and batch_num != 0:
    #                             # print('step')
    #                             # print(parameters['scheduler'].get_last_lr())
    #                             print('line 452')
    #                             print(model + '_lr')
    #                             parameters[model + '_lr'] = scheduler.get_last_lr()[0]
    #                             print(parameters[model + '_lr'])
    #                 else:
    #                     if 'scheduler_interval_batch' in parameters and batch_num % \
    #                             parameters[
    #                                 'scheduler_interval_batch'] == 0 and batch_num != 0:
    #                         # print('step')
    #                         # print(parameters['scheduler'].get_last_lr())
    #                         parameters['lr'] = parameters['scheduler'].get_last_lr()[0]
    #             # if 'scheduler' in parameters and parameters['scheduler'] is not None:
    #             #     parameters['lr'] = parameters['scheduler'].get_last_lr()[0]
    #             functions.add_to_execution_file(parameters,
    #                                             '| epoch {:1d} | btch {:5d} | {:7d}/{:7d}sents(+{:3d}sents) | '
    #                                             'time {:23} | done {:3.1f}% | '
    #                                             'enc.lr {:02.4f} | cla.lr {:02.4f} | ms/batch {:5.2f} | '
    #                                             'cla.loss {:5.2f} | enc.loss {:5.2f} | enc.ppl {:8.2f}'.format(
    #                                                 parameters['epoch'], batch_num, numb_sent,
    #                                                 len(train_data_loader.dataset), prev_sent,
    #                                                 functions.timeSince(start,
    #                                                                     numb_sent / len(train_data_loader.dataset)),
    #                                                 numb_sent / len(train_data_loader.dataset) * 100,
    #                                                 parameters['encoder_lr'], parameters['classifier_lr'],
    #                                                 elapsed * 1000 / parameters['log_interval_batch'],
    #                                                 # Ligne à réfléchir
    #                                                 cls_cur_loss, enc_cur_loss, math.exp(
    #                                                     enc_cur_loss) if enc_cur_loss < 300 else 0))  # math.exp(cur_loss)
    #             # print(target)
    #             # print(output.topk(1))
    #             random_id = random.randint(0, batch[0].size(1) - 1)  # secure the choosen 0 / -1
    #             functions.add_to_execution_file(parameters, str(
    #                 training_functions.sentences_idx_to_word(enc_target.transpose(1, 0),
    #                                                          parameters['embedder'].word2index)[random_id]))
    #             functions.add_to_execution_file(parameters, str(
    #                 training_functions.tensor_to_sentences(output, parameters['embedder'].index2word)[0][random_id]))
    #
    #             prev_sent = numb_sent
    #             if parameters['l1_loss']:
    #                 functions.add_to_execution_file(parameters,
    #                                                 '| F1: {:02.4f} | Precision: {:02.4f} | Recall: {:02.4f}'.format(
    #                                                     f1_score.item(), precision.item(), recall.item()))
    #
    #             enc_total_loss = 0
    #             cls_total_loss = 0
    #             parameters['log_interval_time'] = time.time()
    #         # functions.add_to_execution_file(parameters, 'Fin de log_interval_batch display  ' + str(
    #         #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
    #         # parameters['tmps_form_last_step'] = time.time()
    #     if 'valid_interval_batch' in parameters and batch_num % parameters[
    #         'valid_interval_batch'] == 0 and batch_num != 0:
    #         val_loss = evaluate_seq2seq(parameters, valid_data_loader, save_model=True, end_epoch=False)
    #         # print(parameters['scheduler'].get_last_lr())
    #         # functions.add_to_execution_file(parameters, 'Fin de valid_interval_batch  ' + str(
    #         #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
    #         # parameters['tmps_form_last_step'] = time.time()
    #     else:
    #         a = 1
    # if parameters['scheduler'] is not None:
    #     if type(parameters['scheduler']) is dict:
    #         for model, scheduler in parameters['scheduler'].items():  # créer dictionnaire plutot que liste
    #             print(str(model) + ' ' + str(scheduler.get_last_lr()[0]))
    #             scheduler.step()
    #             parameters[model + '_lr'] = scheduler.get_last_lr()[0]
    #             print(str(model) + ' ' + str(scheduler.get_last_lr()[0]))
    #     else:
    #         print('bad place')
    #         parameters['scheduler'].step()
    # if 'optimizer' in parameters:
    #     if type(parameters['optimizer']) is dict:
    #         for model, optimizer in parameters['optimizer'].items():
    #             optimizer.step()
    # if 'valid_interval_epoch' not in parameters or (
    #         'valid_interval_epoch' in parameters and parameters['epoch'] % parameters[
    #     'valid_interval_epoch'] == 0 and parameters['epoch'] != 0):
    #     val_loss = evaluate_encoder_classifier(parameters, valid_data_loader, save_model=True, end_epoch=True)
    # # scheduler.step()
    # start = time.time()

def linear_int_frontier(parameters, train_data_loader, epsilon = 0.1, max_iter=100):
    search_interval = [0, 1]
    criterion = ClassesDistance()
    diff = float('inf')
    batch_num = 0
    for batch in train_data_loader:  # parameters['batchs']
        batch_num += 1
        iters = 0
        while diff.abs().item() > epsilon and iters < max_iter:
            iters += 1
            input = batch[0]
            target_tensor = batch[1][0]

            encoder_hidden_0, encoder_outputs_0 = parameters['encoder_model'].encoder(input[0], 1, True)
            encoder_hidden_1, encoder_outputs_1 = parameters['encoder_model'].encoder(input[1], 1, True)

            input = (sum(search_interval)/2)*encoder_hidden_0+(1-(sum(search_interval)/2))*encoder_hidden_1
            encoder_outputs = torch.ones_like(encoder_outputs_0)

            output, target, encoder_hidden = parameters['encoder_model'].decode(input, encoder_outputs, 1, target_tensor, False)
            sentence = training_functions.tensor_to_sentences_idx(output)

            value_out, class_out, class_hiddens = parameters['classifier_model'](tuple([sentence, None])) # None ou target c'est pareil car pas utilisé évidément

            diff = criterion(class_out, False)

            print("Différence des probabilités pour le batch "+batch_num+": "+str(diff.abs()))

            # signe_diff = (diff.copy()/diff.copy().abs()).item() # Signe de la différence

            if diff > 0:  # A voir le sens selon comment les batchs sont rangés
                search_interval = [max(search_interval), sum(search_interval)/2]
            elif diff < 0:
                search_interval = [min(search_interval), sum(search_interval)/2]
            else:
                break

        # Enregistrer l'input comme point sur la frontière
        # Voir procédure d'enregistrement et le stockage


#    showPlot(plot_losses)
