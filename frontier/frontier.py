import torch
import torch.functional as F
from torch.nn.modules.loss import _Loss
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer, required
from sklearn.metrics import mean_absolute_error
import time
import training_functions
import pandas as pd


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


def derive_data_from_int_lin(model, batch):
    # Forme du batch

    tuple([torch.cat([batch[0][0], batch[1][0]], dim=0), torch.cat([batch[0][1], batch[1][1]], dim=0)])

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
        print('batch.shape')
        print(batch)
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
        diff = torch.tensor([float('inf')] * batch[0].size(0))
        epsilon = 0.1
        while sum(diff > epsilon).item() != 0 and iter < max_iter:
            iter += 1
            optimizer.zero_grad()
            # print(input.shape)
            output, encoder_hidden, value_out, class_out = parameters['model'](tuple([input, batch[1]]), False)
            diff = criterion(value_out, True)
            diff_soft = criterion(class_out, True)
            # print(diff.grad_fn)
            print(diff_soft.sum())
            print(diff.sum())
            diff.sum().backward()
            optimizer.step()
        else:
            if iter == max_iter:
                print("Le batch " + str(batch_num) + " est arrivé à la limite d'itération fixée à : " + str(max_iter))
            else:
                print("Le batch " + str(batch_num) + " a eu besoin de " + str(iter) + " itérations.")
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


def linear_int_frontier(parameters, train_data_loader, epsilon=0.1, max_iter=100):
    columns = list(range(parameters['model'].encoder.num_layers * parameters['model'].encoder.encode_size))
    columns = ['id'] + columns + ['pred_class_0', 'pred_class_1'] + ['class', 'good_predict', 'frontier']
    # print('columns')
    # print(columns)
    all_points_in_vizu = pd.DataFrame(None, None, columns)
    # print(all_points_in_vizu.head())
    criterion = ClassesDistance()
    batch_num = 0
    for batch in train_data_loader:  # parameters['batchs']
        # print('batch.shape')
        # print(batch)

        # print(torch.cat([batch[0][1].unsqueeze(0), batch[1][1].unsqueeze(0)], dim=0))

        # batch = tuple([torch.cat([batch[0][0].unsqueeze(0), batch[1][0].unsqueeze(0)], dim=0).transpose(1, 0),
        #                torch.cat([batch[0][1].unsqueeze(0), batch[1][1].unsqueeze(0)], dim=0)])

        # print('batch.shape')
        # print(batch)
        diff = torch.tensor(float('inf'))
        search_interval = [0, 1]
        batch_num += 1
        iters = 0
        input = tuple([batch[0][0], batch[1][0]])
        target_tensor = batch[0][0]  # A VOIR

        encoder_hidden_0, encoder_outputs_0 = parameters['encoder_model'].encode(input[0].unsqueeze(1), 1, True)

        encoder_hidden_1, encoder_outputs_1 = parameters['encoder_model'].encode(input[1].unsqueeze(1), 1, True)

        encoder_hidden_0 = torch.cat(
            [tenseur.reshape(
                parameters['encoder_model'].encode_size * parameters['encoder_model'].num_layers).unsqueeze(1) for
             tenseur in
             encoder_hidden_0.unbind(1)], dim=1)  # tenseur, dim=0)
        encoder_hidden_0 = encoder_hidden_0.transpose(1, 0)

        print(encoder_hidden_0)
        print(encoder_hidden_0.type())

        encoder_hidden_1 = torch.cat(
            [tenseur.reshape(
                parameters['encoder_model'].encode_size * parameters['encoder_model'].num_layers).unsqueeze(1) for
             tenseur in
             encoder_hidden_1.unbind(1)], dim=1)  # tenseur, dim=0)
        encoder_hidden_1 = encoder_hidden_1.transpose(1, 0)

        value_out_0 = parameters['classifier_model'](
            encoder_hidden_0)  # None ou target c'est pareil car pas utilisé évidément
        class_out_0 = parameters['model'].sig_out(value_out_0)
        # print(class_out_0)
        value_out_1 = parameters['classifier_model'](
            encoder_hidden_1)  # None ou target c'est pareil car pas utilisé évidément
        class_out_1 = parameters['model'].sig_out(value_out_1)

        while diff.abs().item() > epsilon and iters < max_iter:
            iters += 1

            # print('inputs')
            # print(input[0])
            # print(input[1])



            encoder_hidden = (sum(search_interval) / 2) * encoder_hidden_0 + (
                        1 - (sum(search_interval) / 2)) * encoder_hidden_1
            encoder_outputs = torch.ones_like(encoder_outputs_0)
            # encoder_outputs = torch.zeros_like(encoder_outputs_0)

            # print('input de decode')
            # print(input.shape)

            # output_0, target_0, encoder_hidden_0 = parameters['encoder_model'].decode(new_input, encoder_outputs, 1,
            #                                                                     target_tensor, False)
            # output_1, target_1, encoder_hidden_1 = parameters['encoder_model'].decode(new_input, encoder_outputs, 1,
            #                                                                     target_tensor, False)
            # output, target, encoder_hidden = parameters['encoder_model'].decode(new_input, encoder_outputs, 1,
            #                                                                     target_tensor, False)
            # sentence = training_functions.tensor_to_sentences_idx(output)

            # encoder_hidden = torch.cat(
            #     [tenseur.reshape(
            #         parameters['encoder_model'].encode_size * parameters['encoder_model'].num_layers).unsqueeze(1) for
            #      tenseur in
            #      new_input.unbind(1)], dim=1)  # tenseur, dim=0)
            # encoder_hidden = encoder_hidden.transpose(1, 0)

            # print('input classif')
            # print(tuple([sentence, None]))
            # print(torch.cat(sentence, 1).shape)

            # value_out, class_out, class_hiddens = parameters['classifier_model'](tuple([sentence[0].unsqueeze(1), None])) # None ou target c'est pareil car pas utilisé évidément
            value_out = parameters['classifier_model'](
                encoder_hidden)  # None ou target c'est pareil car pas utilisé évidément
            class_out = parameters['model'].sig_out(value_out)
            diff = criterion(class_out, False)
            # print("Diff " + str(diff.item()) + " output " + str(class_out[0]))
            # print("Différence des probabilités pour le batch " + str(batch_num) + ": " + str(
            #     diff.abs()) + " avec valeur :" + str(sum(search_interval) / 2) + " a l'irer " + str(iters))

            # signe_diff = (diff.copy()/diff.copy().abs()).item() # Signe de la différence
            if diff.abs().item() <= epsilon:
                # print('salut les copains')
                # print(class_out_0)
                all_points_in_vizu = all_points_in_vizu.append(pd.DataFrame([[batch[0][2]] + [element.item() for element in encoder_hidden_0.flatten()] + [class_out_0[0][0].item(), class_out_0[0][1].item()] +
                                          [batch[0][1].item(), batch[0][1].item() == torch.argmax(class_out_0).item(),
                                           False]], None, columns))
                all_points_in_vizu = all_points_in_vizu.append(pd.DataFrame([[batch[1][2]] + [element.item() for element in encoder_hidden_1.flatten()] + [class_out_1[0][0].item(), class_out_1[0][1].item()] +
                                          [batch[1][1].item(), batch[1][1].item() == torch.argmax(class_out_1).item(),
                                           False]], None, columns))
                all_points_in_vizu = all_points_in_vizu.append(pd.DataFrame([[str(batch[0][2])+"_"+str(batch[1][2])] + [element.item() for element in encoder_hidden.flatten()] + [class_out[0][0].item(), class_out[0][1].item()] +
                                          ['frontier', None, True]], None, columns))
                print(str(len(all_points_in_vizu))+" ajoutés")
                # all_points_in_vizu.to_csv("./executions/FRONTIER/all_points.csv", sep=",")
                break
            elif iters == max_iter:
                if sum(search_interval) / 2 <= 0.0001:
                    all_points_in_vizu = all_points_in_vizu.append(pd.DataFrame([[batch[0][2]] + [element.item() for
                                                                                                  element in
                                                                                                  encoder_hidden_0.flatten()] + [
                                                                                     class_out_0[0][0].item(),
                                                                                     class_out_0[0][1].item()] +
                                                                                 [batch[0][1].item(),
                                                                                  batch[0][1].item() == torch.argmax(
                                                                                      class_out_0).item(),
                                                                                  True]], None, columns))
                    all_points_in_vizu = all_points_in_vizu.append(pd.DataFrame([[batch[1][2]] + [element.item() for
                                                                                                  element in
                                                                                                  encoder_hidden_1.flatten()] + [
                                                                                     class_out_1[0][0].item(),
                                                                                     class_out_1[0][1].item()] +
                                                                                 [batch[1][1].item(),
                                                                                  batch[1][1].item() == torch.argmax(
                                                                                      class_out_1).item(),
                                                                                  False]], None, columns))
                elif sum(search_interval) / 2 >= 0.9999:
                    all_points_in_vizu = all_points_in_vizu.append(pd.DataFrame([[batch[0][2]] + [element.item() for
                                                                                                  element in
                                                                                                  encoder_hidden_0.flatten()] + [
                                                                                     class_out_0[0][0].item(),
                                                                                     class_out_0[0][1].item()] +
                                                                                 [batch[0][1].item(),
                                                                                  batch[0][1].item() == torch.argmax(
                                                                                      class_out_0).item(),
                                                                                  False]], None, columns))
                    all_points_in_vizu = all_points_in_vizu.append(pd.DataFrame([[batch[1][2]] + [element.item() for
                                                                                                  element in
                                                                                                  encoder_hidden_1.flatten()] + [
                                                                                     class_out_1[0][0].item(),
                                                                                     class_out_1[0][1].item()] +
                                                                                 [batch[1][1].item(),
                                                                                  batch[1][1].item() == torch.argmax(
                                                                                      class_out_1).item(),
                                                                                  True]], None, columns))
                else:
                    print('Point inclassable')

            else:
                if (batch[0][1].item() == 0 and diff < 0) or (
                        batch[0][1].item() == 1 and diff > 0):  # A voir le sens selon comment les batchs sont rangés
                    search_interval = [max(search_interval), sum(search_interval) / 2]
                elif (batch[0][1].item() == 0 and diff > 0) or (batch[0][1].item() == 1 and diff < 0):
                    search_interval = [min(search_interval), sum(search_interval) / 2]


        # Enregistrer l'input comme point sur la frontière
        # Voir procédure d'enregistrement et le stockage

    all_points_in_vizu.to_csv("./executions/FRONTIER/all_points.csv", sep=",")

def create_all_points_frontier(parameters, train_data_loader, epsilon=0.1, max_iter=100):
    columns = list(range(parameters['model'].encoder.num_layers * parameters['model'].encoder.encode_size))
    columns = ['id'] + columns + ['pred_class_0', 'pred_class_1'] + ['class', 'good_predict', 'frontier']
    # print('columns')
    # print(columns)
    all_points_in_vizu = pd.DataFrame(None, None, columns)
    # print(all_points_in_vizu.head())
    batch_num = 0
    for batch in train_data_loader:  # parameters['batchs']
        if batch_num % 1000 == 0 :
            print("Sentence "+str(batch_num)+" add.")
        # print('batch.shape')
        # print(batch)

        # print(torch.cat([batch[0][1].unsqueeze(0), batch[1][1].unsqueeze(0)], dim=0))

        # batch = tuple([torch.cat([batch[0][0].unsqueeze(0), batch[1][0].unsqueeze(0)], dim=0).transpose(1, 0),
        #                torch.cat([batch[0][1].unsqueeze(0), batch[1][1].unsqueeze(0)], dim=0)])

        # print('batch.shape')
        # print(batch)
        batch_num += 1
        input = tuple([batch[0][0],])
        target_tensor = batch[0][0]  # A VOIR

        encoder_hidden_0, encoder_outputs_0 = parameters['encoder_model'].encode(input[0].unsqueeze(1), 1, True)

        encoder_hidden_0 = torch.cat(
            [tenseur.reshape(
                parameters['encoder_model'].encode_size * parameters['encoder_model'].num_layers).unsqueeze(1) for
             tenseur in
             encoder_hidden_0.unbind(1)], dim=1)  # tenseur, dim=0)
        encoder_hidden_0 = encoder_hidden_0.transpose(1, 0)


        value_out_0 = parameters['classifier_model'](
            encoder_hidden_0)  # None ou target c'est pareil car pas utilisé évidément
        class_out_0 = parameters['model'].sig_out(value_out_0)


        all_points_in_vizu = all_points_in_vizu.append(pd.DataFrame([[batch[0][2]] + [element.item() for element
                                                                                      in
                                                                                      encoder_hidden_0.flatten()] + [
                                                                         class_out_0[0][0].item(),
                                                                         class_out_0[0][1].item()] +
                                                                     [batch[0][1].item(),
                                                                      batch[0][1].item() == torch.argmax(
                                                                          class_out_0).item(),
                                                                      False]], None, columns))

        # Enregistrer l'input comme point sur la frontière
        # Voir procédure d'enregistrement et le stockage

    all_points_in_vizu.to_csv("./executions/FRONTIER/very_all_points.csv", sep=",")

def create_all_points_from_nn_frontier(parameters, file, epsilon=0.1, max_iter=100):
    # A RENDRE POSSIBLE POUR UN BATCH ENTIER

    data = pd.read_csv(file)
    data = data.drop(['Unnamed: 0'], axis=1)
    columns = list(data.columns)+['value_0', 'value_1']
    data_frontier = pd.DataFrame(data=None, columns=columns)

    print(data)
    print(data_frontier.head())

    parameters['model'].eval()

    # columns = data.columns

    criterion = ClassesDistance()

    batch_num = 0
    for point in data.index:  # parameters['batchs']
        if batch_num % 10000 == 0 and batch_num != 0:
            print("Sentence "+str(batch_num)+" add.")
            data_frontier.to_csv("./executions/FRONTIER/nn_frontier_V3_provisoire_"+str(batch_num)+".csv", sep=",")

        batch_num += 1
        iters = 0
        diff = torch.tensor(float('inf'))

        search_interval = [0, 1]

        # print(point)
        id_point = data['id'][point]
        id_nearest = data[data['id'] == id_point]['nearest'].values[0]
        # print(id_nearest)
        id_frontier = str(id_point) + '_' + str(id_nearest)

        # print(data[data['id'] == id_point].iloc[:, 1:1025].to_numpy()) # 1025 a remplacer

        # print(data[data['id'] == id_nearest])

        encoder_hidden_0 = torch.from_numpy(data[data['id'] == id_point].iloc[:, 1:1025].to_numpy()).type(torch.FloatTensor).to(parameters['device'])
        encoder_hidden_1 = torch.from_numpy(data[data['id'] == id_nearest].iloc[:, 1:1025].to_numpy()).type(torch.FloatTensor).to(parameters['device'])
        # print(encoder_hidden_1)
        encoder_hidden_0.requires_grad = True
        encoder_hidden_1.requires_grad = True

        # print(encoder_hidden_0)
        # print(encoder_hidden_0.type())

        value_out_0 = parameters['classifier_model'](
            encoder_hidden_0)  # None ou target c'est pareil car pas utilisé évidément
        class_out_0 = parameters['model'].sig_out(value_out_0)

        value_out_1 = parameters['classifier_model'](
            encoder_hidden_1)  # None ou target c'est pareil car pas utilisé évidément
        class_out_1 = parameters['model'].sig_out(value_out_1)

        while diff.abs().item() > epsilon and iters < max_iter:
            iters += 1

            # print('inputs')
            # print(input[0])
            # print(input[1])

            # print(encoder_hidden_0)
            # print(encoder_hidden_1)

            encoder_hidden = (sum(search_interval) / 2) * encoder_hidden_0 + (
                    1 - (sum(search_interval) / 2)) * encoder_hidden_1
            # print(encoder_hidden)
            # value_out, class_out, class_hiddens = parameters['classifier_model'](tuple([sentence[0].unsqueeze(1), None])) # None ou target c'est pareil car pas utilisé évidément
            value_out = parameters['classifier_model'](
                encoder_hidden)  # None ou target c'est pareil car pas utilisé évidément
            class_out = parameters['model'].sig_out(value_out)
            # print(class_out)
            diff = criterion(class_out, False)
            # print("Diff " + str(diff.item()) + " output " + str(class_out[0]))
            # print("Différence des probabilités pour le batch " + str(batch_num) + ": " + str(
            #     diff.abs()) + " avec valeur :" + str(sum(search_interval) / 2) + " a l'irer " + str(iters))

            # signe_diff = (diff.copy()/diff.copy().abs()).item() # Signe de la différence
            # print(diff)
            if diff.abs().item() <= epsilon:
                # print('salut les copains')
                # print(class_out_0)

                data_frontier = data_frontier.append(
                    pd.DataFrame([[id_frontier] + [element.item() for element in encoder_hidden.flatten()] + [
                                                                                 class_out[0][0].item(),
                                                                                 class_out[0][1].item()] +
                                                                             ['frontier', None, True, None]+[value_out[0][0].item(), value_out[0][1].item()]], None,
                                                                            columns))
                data_frontier = data_frontier.append(
                    pd.DataFrame([[id_point] + [element.item() for element in encoder_hidden_0.flatten()] + [
                    class_out_0[0][0].item(),
                    class_out_0[0][1].item()] +
                              [data[data['id'] == id_point]['class'].values[0],
                               data[data['id'] == id_point]['good_predict'].values[0],
                               False, id_nearest] + [value_out_0[0][0].item(), value_out_0[0][1].item()]], None,
                             columns))

                # data_frontier = data_frontier.append(
                #     pd.DataFrame([[id_point] + [element.item() for element in encoder_hidden_0.flatten()] + [
                #     class_out_1[0][0].item(),
                #     class_out_1[0][1].item()] +
                #               [data[data['id'] == id_point]['class'].values[0],
                #                data[data['id'] == id_point]['good_predict'].values[0],
                #                False, id_nearest] + [value_out_1[0][0].item(), value_out_1[0][1].item()]], None,
                #              columns))

                print(str(len(data_frontier)) + " ajoutés")
                # all_points_in_vizu.to_csv("./executions/FRONTIER/all_points.csv", sep=",")
                break
            elif iters == max_iter:
                # if sum(search_interval) / 2 <= 0.0001:
                #     data_frontier = data_frontier.append(
                #         pd.DataFrame([[id_point] + [element.item() for element in encoder_hidden_0.flatten()] + [
                #                                                                      class_out_0[0][0].item(),
                #                                                                      class_out_0[0][1].item()] +
                #                                                                  [data[data['id'] == id_point]['class'].values[0],
                #                                                                   data[data['id'] == id_point]['good_predict'].values[0],
                #                                                                   True, id_nearest]+[value_out_0[0][0].item(), value_out_0[0][1].item()]], None, columns))

                    # data_frontier = data_frontier.append(pd.DataFrame([[id_nearest] + [element.item() for
                    #                                                                               element in
                    #                                                                               encoder_hidden_1.flatten()] + [
                    #                                                                  class_out_1[0][0].item(),
                    #                                                                  class_out_1[0][1].item()] +
                    #                                                              [data[data['id'] == id_nearest]['class'].values[0],
                    #                                                               data[data['id'] == id_nearest]['good_predict'].values[0],
                    #                                                               False, data[data['id'] == id_nearest]['nearest'].values[0]] +
                    #                                                    [value_out_1[0][0].item(), value_out_1[0][1].item()]], None, columns))
                # elif sum(search_interval) / 2 >= 0.9999:
                    data_frontier = data_frontier.append(pd.DataFrame([[id_point] + [element.item() for
                                                                                                  element in
                                                                                                  encoder_hidden_0.flatten()] + [
                                                                                     class_out_0[0][0].item(),
                                                                                     class_out_0[0][1].item()] +
                                                                                 [data[data['id'] == id_point][
                                                                                      'class'].values[0],
                                                                                  data[data['id'] == id_point][
                                                                                      'good_predict'].values[0],
                                                                                  False, id_nearest]+[value_out_0[0][0].item(), value_out_0[0][1].item()]], None, columns))
                    # data_frontier = data_frontier.append(pd.DataFrame([[id_nearest] + [element.item() for
                    #                                                                               element in
                    #                                                                               encoder_hidden_1.flatten()] + [
                    #                                                                  class_out_1[0][0].item(),
                    #                                                                  class_out_1[0][1].item()] +
                    #                                                              [data[data['id'] == id_nearest][
                    #                                                                   'class'].values[0],
                    #                                                               data[data['id'] == id_nearest][
                    #                                                                   'good_predict'].values[0],
                    #                                                               True, data[data['id'] == id_nearest]['nearest'].values[0]]
                    #                                                    +[value_out_1[0][0].item(), value_out_1[0][1].item()]], None, columns))
                # else:
                #     print('Point inclassable')

            else:
                if (data[data['id'] == id_point]['class'].values[0] == 0 and diff < 0) or (
                        data[data['id'] == id_point]['class'].values[0] == 1 and diff > 0):  # A voir le sens selon comment les batchs sont rangés
                    # print(data[data['id'] == id_point]['class'])
                    # print('là')
                    search_interval = [max(search_interval), sum(search_interval) / 2]
                elif (data[data['id'] == id_point]['class'].values[0] == 0 and diff > 0) or (
                        data[data['id'] == id_point]['class'].values[0] == 1 and diff < 0):
                    # print(data[data['id'] == id_point]['class'])
                    # print('ici')
                    search_interval = [min(search_interval), sum(search_interval) / 2]
        # print('inputs')
        # print(input[0])
        # print(input[1])


        # Enregistrer l'input comme point sur la frontière
        # Voir procédure d'enregistrement et le stockage

    data_frontier.to_csv("./executions/FRONTIER/nn_frontier_V3.csv", sep=",")

