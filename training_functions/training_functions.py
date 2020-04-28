import time
import functions
import math
import torch
import os
import glob


# Code from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# Change : Yes

def full_train(parameters, train_data_loader, valid_data_loader):
    scheduler = parameters['scheduler']
    parameters['best_val_loss'] = float("inf")
    parameters['epoch'] = 0
    f1_loss = functions.F1_Loss_Sentences(parameters['model_params']['ntoken'], parameters['device'])
    # functions.add_to_execution_file(parameters, 'Fin de creation des parametres train en  ' + str(
    #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
    # parameters['tmps_form_last_step'] = time.time()
    while parameters['epoch'] < parameters['epochs']:
        parameters['epoch_start_time'] = time.time()
        parameters['model'].train()  # Turn on the train mode
        total_loss = 0.
        parameters['log_interval_time'] = time.time()
        # ntokens = len(parameters['embedder'].index2word)
        parameters['epoch'] += 1
        batch_num = 0
        for batch in train_data_loader:  # parameters['batchs']
            parameters['batch_start_time'] = time.time()
            # print(batch[0].shape)
            # print(batch[1].shape)
            # data, targets = get_batch(train_data, i)
            parameters['optimizer'].zero_grad()
            # functions.add_to_execution_file(parameters, 'Fin deprocedure avant Nnetwork en  ' + str(
            #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
            # parameters['tmps_form_last_step'] = time.time()
            output, target = parameters['model'](batch)  # écrire function one_train
            # functions.add_to_execution_file(parameters, 'Fin du passage en Nnetwork en  ' + str(
            #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
            # parameters['tmps_form_last_step'] = time.time()
            # print(output.device)
            if parameters['l1_loss']:
                f1_score, precision, recall = f1_loss(output, target)
            # functions.add_to_execution_file(parameters, 'Fin du calcul du F1  ' + str(
            #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
            # parameters['tmps_form_last_step'] = time.time()
            # print(f1_score.item())
            # print(precision.item())
            # print(recall.item())
            # print(output.shape)
            # print(target.shape)
            loss = parameters['criterion'](output, target)
            # functions.add_to_execution_file(parameters, 'Fin de calcul de loss  ' + str(
            #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
            # parameters['tmps_form_last_step'] = time.time()
            loss.backward()
            if 'grad_norm' in parameters and parameters['grad_norm']:
                parameters.grad_norm(parameters['model'].parameters())
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                # functions.add_to_execution_file(parameters, 'Fin de grad_norm  ' + str(
                #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
                # parameters['tmps_form_last_step'] = time.time()
            if 'optimizer' in parameters:
                parameters['optimizer'].step()
                # functions.add_to_execution_file(parameters, 'Fin de optimizer  ' + str(
                #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
                # parameters['tmps_form_last_step'] = time.time()
            '''for name, param in parameters['model'].named_parameters():
                if param.requires_grad:
                    print(name, param.data)'''

            # print(target)
            # print(loss)
            total_loss += loss.item()
            # print(total_loss)
            if 'log_interval_batch' in parameters:
                # print(batch_num % parameters['log_interval_batch'])
                # print(batch_num)
                if batch_num % parameters['log_interval_batch'] == 0 and batch_num > 0:
                    # batch.num doit exister voir dataloader !!!
                    cur_loss = total_loss / parameters['log_interval_batch']
                    elapsed = time.time() - parameters['log_interval_time']
                    functions.add_to_execution_file(parameters, '| epoch {:3d} | {:5d}/{:5d} sents | '
                                                                'lr {:02.4f} | ms/batch {:5.2f} | '
                                                                'loss {:5.2f} | ppl {:8.2f}'.format(
                        parameters['epoch'], batch_num, len(train_data_loader), scheduler.get_last_lr()[0],
                        elapsed * 1000 / parameters['log_interval_batch'],  # Ligne à réfléchir
                        cur_loss, math.exp(cur_loss)))
                    if parameters['l1_loss']:
                        functions.add_to_execution_file(parameters,
                                                        '| F1: {:02.4f} | Precision: {:02.4f} | Recall: {:02.4f}'.format(
                                                            f1_score.item(), precision.item(), recall.item()))

                    total_loss = 0
                    parameters['log_interval_time'] = time.time()
                # functions.add_to_execution_file(parameters, 'Fin de log_interval_batch display  ' + str(
                #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
                # parameters['tmps_form_last_step'] = time.time()
            if 'valid_interval_batch' in parameters and batch_num % parameters[
                'valid_interval_batch'] == 0 and batch_num != 0:
                val_loss = evaluate(parameters, valid_data_loader, save_model=True, end_epoch=False)
                if 'scheduler_interval_batch' in parameters and parameters['scheduler_interval_batch']:
                    scheduler.step()
                # functions.add_to_execution_file(parameters, 'Fin de valid_interval_batch  ' + str(
                #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
                # parameters['tmps_form_last_step'] = time.time()
            else:
                a = 1
            batch_num += 1

        val_loss = evaluate(parameters, valid_data_loader, save_model=True, end_epoch=True)
        scheduler.step()


def evaluate(parameters, valid_data_loader, save_model=False, end_epoch=False):
    start_time = time.time()
    parameters['model'].eval()  # Turn on the evaluation mode
    valid_total_loss = 0.
    # ntokens = len(parameters['embedder'].index2word)
    with torch.no_grad():
        for batch in valid_data_loader:  # parameters['batchs']
            # data, targets = get_batch(data_source, i)
            output, target = parameters['model'](batch)  # écrire function one_train
            valid_loss = parameters['criterion'](output, target)
            valid_total_loss += valid_loss.item()

    val_loss = valid_total_loss / (len(valid_data_loader))  # valid_total_loss / (len(valid_data_loader) - 1)
    functions.add_to_execution_file(parameters, '-' * 89)
    if end_epoch:
        functions.add_to_execution_file(parameters, '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                                                    'valid ppl {:8.2f}'.format(parameters['epoch'],
                                                                               (time.time() - start_time),
                                                                               val_loss, math.exp(val_loss)))
    else:
        functions.add_to_execution_file(parameters, '| inside epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                                                    'valid ppl {:8.2f}'.format(parameters['epoch'],
                                                                               (time.time() - start_time),
                                                                               val_loss, math.exp(val_loss)))
    functions.add_to_execution_file(parameters, '-' * 89)
    if save_model:
        best_model_and_save(parameters, val_loss)
    return val_loss


def best_model_and_save(parameters, val_loss):
    try:
        os.makedirs("./executions/" + parameters['execution_name'] + "/models/")
        functions.add_to_execution_file(parameters,
                                        "Directory " + parameters['execution_name'] + "/models/" + " Created ")
    except FileExistsError:
        functions.add_to_execution_file(parameters,
                                        "Directory " + parameters['execution_name'] + "/models/" + " already exists")
    if val_loss < parameters['best_val_loss']:
        parameters['best_val_loss'] = val_loss
        parameters['best_model'] = parameters['model']
        for f in glob.glob("./executions/" + str(parameters['execution_name']) + "/models/Best_Model_Epoch*.pt"):
            os.remove(f)
        torch.save(parameters['best_model'].to('cpu').state_dict(),
                   "./executions/" + str(parameters['execution_name']) + "/models/CPU_Best_Model_Epoch_" + str(
                       parameters['epoch']) + ".pt")
        torch.save(parameters['best_model'].to(parameters['device']).state_dict(),
                   "./executions/" + str(parameters['execution_name']) + "/models/Best_Model_Epoch_" + str(
                       parameters['epoch']) + ".pt")
    torch.save(parameters['model'].state_dict(),
               "./executions/" + str(parameters['execution_name']) + "/models/Model_Epoch_" + str(
                   parameters['epoch']) + ".pt")


def autoencoder_seq2seq_train(parameters, train_data_loader, valid_data_loader):
    start = time.time()
    parameters['best_val_loss'] = float("inf")
    parameters['epoch'] = 0
    # f1_loss = functions.F1_Loss_Sentences(parameters['model_params']['ntoken'], parameters['device'])
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_every = 100
    plot_every = 5000

    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # training_pairs = [tensorsFromPair(random.choice(pairs))
    #                  for i in range(n_iters)]
    # criterion = nn.NLLLoss()
    while parameters['epoch'] < parameters['epochs']:
        parameters['epoch_start_time'] = time.time()
        # parameters['model'].train()  # Turn on the train mode
        total_loss = 0.
        parameters['log_interval_time'] = time.time()
        # ntokens = len(parameters['embedder'].index2word)
        parameters['epoch'] += 1
        batch_num = 0
        numb_sent = 0
        #print(len(train_data_loader.dataset))
        for batch in train_data_loader:  # parameters['batchs']
            # print('debut du batch')
            # print(batch)
            # print(len(train_data_loader[batch[0]][0]))
            # print(len(train_data_loader[batch[1]][0]))
            # batch = parameters['collate_fn']([train_data_loader[i] for i in batch])
            parameters['model'].train()
            numb_sent += batch[0].size(1)
            batch_num +=1
            parameters['batch_start_time'] = time.time()
            parameters['optimizer'].zero_grad()
            # print(batch_num)
            loss = 0
            # print('batch structure')
            # print(batch[0].shape)
            # print(batch[1].shape)
            output, target = parameters['model'](batch)
            # print('model output')
            # print(output.shape)
            # print(target.shape)
            for di in range(len(output)):
                # print('model output di')
                # print(output.shape)
                # print(target[di].shape)
                loss += parameters['criterion'](output[di], target[di])  # voir pourquoi unsqueeze
            loss.backward()
            if 'grad_norm' in parameters and parameters['grad_norm']:
                parameters.grad_norm(parameters['model'].parameters())
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                # functions.add_to_execution_file(parameters, 'Fin de grad_norm  ' + str(
                #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
                # parameters['tmps_form_last_step'] = time.time()
            if parameters['scheduler'] is not None and 'scheduler_interval_batch' in parameters and batch_num % \
                    parameters[
                        'scheduler_interval_batch'] == 0 and batch_num != 0:
                # print('step')
                # print(parameters['scheduler'].get_last_lr())
                parameters['scheduler'].step()
            if 'optimizer' in parameters:
                # print(parameters['scheduler'].get_last_lr())
                parameters['optimizer'].step()
                # print(parameters['scheduler'].get_last_lr())
                # functions.add_to_execution_file(parameters, 'Fin de optimizer  ' + str(
                #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
                # parameters['tmps_form_last_step'] = time.time()
            '''for name, param in parameters['model'].named_parameters():
                if param.requires_grad:
                    print(name, param.data)'''

            # print(target)
            # print(loss)
            total_loss += loss.item()
            # print(batch_num)
            if 'log_interval_batch' in parameters and batch_num % parameters[
                'log_interval_batch'] == 0 and batch_num > 0:
                # print(batch_num % parameters['log_interval_batch'])
                # print(batch_num)
                if batch_num % parameters['log_interval_batch'] == 0 and batch_num > 0:
                    # batch.num doit exister voir dataloader !!!
                    cur_loss = total_loss / parameters['log_interval_batch']
                    elapsed = time.time() - parameters['log_interval_time']
                    if 'scheduler' in parameters and parameters['scheduler'] is not None:
                        parameters['lr'] = parameters['scheduler'].get_last_lr()[0]
                    functions.add_to_execution_file(parameters, '| epoch {:3d} | {:7d}/{:7d}sents | '
                                                                'time {:23} | done {:3.1f}% | '
                                                                'lr {:02.4f} | ms/batch {:5.2f} | '
                                                                'loss {:5.2f} | ppl {:8.2f}'.format(
                        parameters['epoch'], numb_sent, len(train_data_loader.dataset),
                        functions.timeSince(start, numb_sent / len(train_data_loader.dataset)), numb_sent / len(train_data_loader.dataset) * 100,
                        parameters['lr'], elapsed * 1000 / parameters['log_interval_batch'],  # Ligne à réfléchir
                        cur_loss, math.exp(cur_loss) if cur_loss < 300 else 0))  # math.exp(cur_loss)
                    if parameters['l1_loss']:
                        functions.add_to_execution_file(parameters,
                                                        '| F1: {:02.4f} | Precision: {:02.4f} | Recall: {:02.4f}'.format(
                                                            f1_score.item(), precision.item(), recall.item()))

                    total_loss = 0
                    parameters['log_interval_time'] = time.time()
                # functions.add_to_execution_file(parameters, 'Fin de log_interval_batch display  ' + str(
                #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
                # parameters['tmps_form_last_step'] = time.time()
            if 'valid_interval_batch' in parameters and batch_num % parameters[
                'valid_interval_batch'] == 0 and batch_num != 0:
                val_loss = evaluate_seq2seq(parameters, valid_data_loader, save_model=True, end_epoch=False)
                # print(parameters['scheduler'].get_last_lr())
                # functions.add_to_execution_file(parameters, 'Fin de valid_interval_batch  ' + str(
                #     round(time.time() - parameters['tmps_form_last_step']),2) + ' secondes')
                # parameters['tmps_form_last_step'] = time.time()
            else:
                a = 1

        val_loss = evaluate_seq2seq(parameters, valid_data_loader, save_model=True, end_epoch=True)
        # scheduler.step()


'''            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0'''


#    showPlot(plot_losses)

def evaluate_seq2seq(parameters, valid_data_loader, save_model=False, end_epoch=False):
    start_time = time.time()
    parameters['model'].eval()  # Turn on the evaluation mode
    valid_total_loss = 0.
    # ntokens = len(parameters['embedder'].index2word)
    print('debut val')
    with torch.no_grad():
        for batch in valid_data_loader:  # parameters['batchs']
            # data, targets = get_batch(data_source, i)
            output, target = parameters['model'](batch)  # écrire function one_train
            # valid_loss = parameters['criterion'](output, target)
            for di in range(len(output)):
                # print('model output di')
                # print(output.shape)
                # print(target[di].shape)
                valid_total_loss += parameters['criterion'](output[di], target[di])  # voir pourquoi unsqueeze

    val_loss = valid_total_loss / (len(valid_data_loader))  # valid_total_loss / (len(valid_data_loader) - 1)
    functions.add_to_execution_file(parameters, '-' * 89)
    if end_epoch:
        functions.add_to_execution_file(parameters, '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                                                    'valid ppl {:8.2f}'.format(parameters['epoch'],
                                                                               (time.time() - start_time),
                                                                               val_loss, math.exp(
                val_loss) if val_loss < 300 else 0))
    else:
        functions.add_to_execution_file(parameters, '| inside epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                                                    'valid ppl {:8.2f}'.format(parameters['epoch'],
                                                                               (time.time() - start_time),
                                                                               val_loss, math.exp(
                val_loss) if val_loss < 300 else 0))
    functions.add_to_execution_file(parameters, '-' * 89)
    if save_model:
        best_model_and_save(parameters, val_loss)
    return val_loss


def one_train(bacth, parameters):
    # decoder_optimizer.zero_grad()

    loss = 0

    decoder_output, target_tensor = parameters['model'](batch)
    for di in range(len(decoder_output)):
        loss += parameters['criterion'](decoder_output[di], target_tensor[di])

    loss.backward()

    parameters['optimizer'].step()
    # parameters['encoder_optimizer'].step()
    # parameters['decoder_optimizer'].step()

    return loss.item() / target_length

    encoder_hidden = model.encoder.initHidden()
    parameters['optimizer'].zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
