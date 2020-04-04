import time
import functions
import math
import torch


# Code from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# Change : Yes

def full_train(parameters, train_data_loader, valid_data_loader, one_train):
    criterion = parameters['criterion']  # useless
    lr = parameters['lr']  # learning rate
    optimizer = parameters['optimizer']
    scheduler = parameters['scheduler']
    best_val_loss = float("inf")

    for epoch in range(1, parameters['epochs'] + 1):
        epoch_start_time = time.time()

        parameters['model'].train()  # Turn on the train mode

        total_loss = 0.
        start_time = time.time()
        ntokens = len(parameters['embedder'].index2word)
        epoch = 0
        while epoch < parameters['epochs']:
            epoch += 1
            batch_num = 0
            for batch in train_data_loader:  # parameters['batchs']
                # data, targets = get_batch(train_data, i)
                parameters['optimizer'].zero_grad()
                #print(batch)
                output, target = parameters['model'](batch)  # écrire function one_train
                #print(output.shape)
                #print(target.shape)
                loss = parameters['criterion'](output, target)
                loss.backward()
                if 'grad_norm' in parameters and parameters['grad_norm']:
                    parameters.grad_norm(parameters['model'].parameters())
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                if 'optimizer' in parameters and parameters['optimizer']:
                    parameters['optimizer'].step()

                total_loss += loss.item()
                if 'log_interval_batch' in parameters:
                    #print(batch_num % parameters['log_interval_batch'])
                    #print(batch_num)
                    if batch_num % parameters['log_interval_batch'] == 0 and batch_num > 0:
                        # batch.num doit exister voir dataloader !!!
                        cur_loss = total_loss / parameters['log_interval_batch']
                        elapsed = time.time() - start_time
                        functions.add_to_execution_file(parameters, '| epoch {:3d} | {:5d}/{:5d} batches | '
                                                        'lr {:02.2f} | ms/batch {:5.2f} | '
                                                        'loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, batch_num, len(train_data_loader), scheduler.get_lr()[0],
                            elapsed * 1000 / parameters['log_interval_batch'],
                            cur_loss, math.exp(cur_loss)))
                        total_loss = 0
                        start_time = time.time()
                else:
                    a = 1
                batch_num += 1

        val_loss = evaluate(parameters['model'], valid_data_loader)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()


def evaluate(eval_model, data_source, parameters):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(parameters['embedder'].index2word)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)
