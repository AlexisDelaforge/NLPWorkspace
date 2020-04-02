import time
import functions


# Code from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# Change : Yes

def full_train(parameters, train_data_loader, valid_data_loader, one_train):
    criterion = parameters['criterion']  # useless
    lr = parameters['lr']  # learning rate
    optimizer = parameters['optimizer']
    scheduler = parameters['scheduler'](**functions.dict_less(parameters['scheduler_params'], ['optimizer']))

    parameters['model'].train()  # Turn on the train mode

    total_loss = 0.
    start_time = time.time()
    # ntokens = len(TEXT.vocab.stoi)
    epoch = 0
    while epoch < parameters['epochs']:
        epoch += 1
        for batch in parameters['batchs']:
            # data, targets = get_batch(train_data, i)
            parameters['optimizer'].zero_grad()
            output, target = parameters['one_train'](batch)
            loss = parameters['criterion'](output, target)
            loss.backward()
            if 'grad_norm' in parameters:
                parameters.grad_norm(
                    parameters['model'].parameters())  # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if 'optimizer' in parameters:
                parameters['optimizer'].step()

            total_loss += loss.item()
            if 'log_interval_batch' in parameters:
                if batch.num % parameters['log_interval_batch'] == 0 and batch > 0:  # batch.num doit exister voir dataloader !!!
                    cur_loss = total_loss / parameters['log_interval_batch']
                    elapsed = time.time() - start_time
                    functions.add_to_execution_file('| epoch {:3d} | {:5d}/{:5d} batches | '
                                                    'lr {:02.2f} | ms/batch {:5.2f} | '
                                                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch.num, parameters['batchs'] // bptt, scheduler.get_lr()[0],
                                          elapsed * 1000 / parameters['log_interval_batch'],
                        cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    start_time = time.time()
            else:
                a = 1


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)
