import time
import functions
import math
import torch
import os
import glob

# Code from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# Change : Yes

def full_train(parameters, train_data_loader, valid_data_loader, one_train):
    scheduler = parameters['scheduler']
    parameters['best_val_loss'] = float("inf")
    parameters['epoch'] = 0
    f1_loss = functions.F1_Loss_Sentences(parameters['model_params']['ntoken']).to(parameters['device'])
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
            # data, targets = get_batch(train_data, i)
            parameters['optimizer'].zero_grad()
            #print(batch)
            output, target = parameters['model'](batch)  # écrire function one_train
            f1_score, precision, recall = f1_loss(output, target)
            #print(f1_score.item())
            #print(precision.item())
            #print(recall.item())
            #print(output.shape)
            #print(target.shape)
            loss = parameters['criterion'](output, target)
            loss.backward()
            if 'grad_norm' in parameters and parameters['grad_norm']:
                parameters.grad_norm(parameters['model'].parameters())
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if 'optimizer' in parameters and parameters['optimizer']:
                parameters['optimizer'].step()
            '''for name, param in parameters['model'].named_parameters():
                if param.requires_grad:
                    print(name, param.data)'''
            #print(target)
            #print(loss)
            total_loss += loss.item()
            #print(total_loss)
            if 'log_interval_batch' in parameters:
                #print(batch_num % parameters['log_interval_batch'])
                #print(batch_num)
                if batch_num % parameters['log_interval_batch'] == 0 and batch_num > 0:
                    # batch.num doit exister voir dataloader !!!
                    cur_loss = total_loss / parameters['log_interval_batch']
                    elapsed = time.time() - parameters['log_interval_time']
                    functions.add_to_execution_file(parameters, '| epoch {:3d} | {:5d}/{:5d} batches | '
                                                    'lr {:02.4f} | ms/batch {:5.2f} | '
                                                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        parameters['epoch'], batch_num, len(train_data_loader), scheduler.get_lr()[0],
                        elapsed * 1000 / parameters['log_interval_batch'], # Ligne à réfléchir
                        cur_loss, math.exp(cur_loss)))
                    functions.add_to_execution_file(parameters, '| F1: {:02.4f} | Precision: {:02.4f} | Recall: {:02.4f}'.format(f1_score.item(), precision.item(), recall.item()))

                    total_loss = 0
                    parameters['log_interval_time'] = time.time()
            if 'valid_interval_batch' in parameters and batch_num % parameters['valid_interval_batch'] == 0 and batch_num != 0:
                val_loss = evaluate(parameters, valid_data_loader, save_model = True, end_epoch = False)
                if 'scheduler_interval_batch' in parameters and parameters['scheduler_interval_batch']:
                    scheduler.step()
            else:
                a = 1
            batch_num += 1


        val_loss = evaluate(parameters, valid_data_loader, save_model = True, end_epoch = True)
        scheduler.step()


def evaluate(parameters, valid_data_loader, save_model = False, end_epoch = False):
    start_time = time.time()
    parameters['model'].eval()  # Turn on the evaluation mode
    valid_total_loss = 0.
    # ntokens = len(parameters['embedder'].index2word)
    with torch.no_grad():
        for batch in valid_data_loader:  # parameters['batchs']
            #data, targets = get_batch(data_source, i)
            output, target = parameters['model'](batch)  # écrire function one_train
            valid_loss = parameters['criterion'](output, target)
            valid_total_loss += valid_loss.item()

    val_loss = valid_total_loss / (len(valid_data_loader))  # valid_total_loss / (len(valid_data_loader) - 1)
    print('-' * 89)
    if end_epoch:
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(parameters['epoch'], (time.time() - start_time),
                                     val_loss, math.exp(val_loss)))
    else:
        print('| inside epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(parameters['epoch'], (time.time() - start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)
    if save_model:
        best_model_and_save(parameters, val_loss)
    return val_loss

def best_model_and_save(parameters, val_loss):
    try:
        os.makedirs("./executions/" + parameters['execution_name'] + "/models/")
        print("Directory ", parameters['execution_name'] + "/models/", " Created ")
    except FileExistsError:
        print("Directory ", parameters['execution_name'] + "/models/", " already exists")
    if val_loss < parameters['best_val_loss']:
        parameters['best_val_loss'] = val_loss
        parameters['best_model'] = parameters['model']
        for f in glob.glob("./executions/" + str(parameters['execution_name']) + "/models/Best_Model_Epoch*.pt"):
            os.remove(f)
        torch.save(parameters['best_model'].state_dict(),"./executions/" + str(parameters['execution_name']) + "/models/Best_Model_Epoch_" + str(parameters['epoch'])+".pt")
    torch.save(parameters['model'].state_dict(),"./executions/" + str(parameters['execution_name']) + "/models/Model_Epoch_" + str(parameters['epoch']) + ".pt")
