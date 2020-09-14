import copy
import torch
import torch.nn as nn
import dsl


from utils.data import pad_minibatch, unpad_minibatch, flatten_tensor, flatten_batch
from utils.logging import log_and_print


def init_optimizer(program, optimizer, lr):
    queue = [program]
    all_params = []
    while len(queue) != 0:
        current_function = queue.pop()
        if issubclass(type(current_function), dsl.HeuristicNeuralFunction):
            current_function.init_model()
            all_params.append({'params' : current_function.model.parameters(),'lr' : lr})
        elif current_function.has_params:
            current_function.init_params()
            all_params.append({'params': list(current_function.parameters.values()), 'lr': lr})
        else:
            for submodule, functionclass in current_function.submodules.items():
                queue.append(functionclass)
    curr_optim = optimizer(all_params, lr)
    return curr_optim

def process_batch(program, batch, output_type, output_size, device='cpu'):
    batch_input = [torch.tensor(traj) for traj in batch]
    batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
    batch_padded = batch_padded.to(device)
    out_padded = program.execute_on_batch(batch_padded, batch_lens)
    if output_type == "list":
        out_unpadded = unpad_minibatch(out_padded, batch_lens, listtoatom=(program.output_type=='atom'))
    else:
        out_unpadded = out_padded
    if output_size == 1 or output_type == "list":
        return flatten_tensor(out_unpadded).squeeze()
    else:
        if isinstance(out_unpadded, list):
            out_unpadded = torch.cat(out_unpadded, dim=0).to(device)          
        return out_unpadded

def execute_and_train(program, validset, trainset, train_config, output_type, output_size, 
    neural=False, device='cpu', use_valid_score=False, print_every=60):

    lr = train_config['lr']
    neural_epochs = train_config['neural_epochs']
    symbolic_epochs = train_config['symbolic_epochs']
    optimizer = train_config['optimizer']
    lossfxn = train_config['lossfxn']
    evalfxn = train_config['evalfxn']
    num_labels = train_config['num_labels']

    num_epochs = neural_epochs if neural else symbolic_epochs

    # initialize optimizer
    curr_optim = init_optimizer(program, optimizer, lr)

    # prepare validation set
    validation_input, validation_output = map(list, zip(*validset))
    validation_true_vals = torch.tensor(flatten_batch(validation_output)).float().to(device)
    # TODO a little hacky, but easiest solution for now
    if isinstance(lossfxn, nn.CrossEntropyLoss):
        validation_true_vals = validation_true_vals.long()

    best_program = None
    best_metric = float('inf')
    best_additional_params = {}

    for epoch in range(1, num_epochs+1):
        for batchidx in range(len(trainset)):
            batch_input, batch_output = map(list, zip(*trainset[batchidx]))
            true_vals = torch.tensor(flatten_batch(batch_output)).float().to(device)
            predicted_vals = process_batch(program, batch_input, output_type, output_size, device)
            # TODO a little hacky, but easiest solution for now
            if isinstance(lossfxn, nn.CrossEntropyLoss):
                true_vals = true_vals.long()
            #print(predicted_vals.shape, true_vals.shape)
            loss = lossfxn(predicted_vals, true_vals)
            curr_optim.zero_grad()
            loss.backward()
            curr_optim.step()

            # if batchidx % print_every == 0 or batchidx == 0:
            #     log_and_print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, loss.item()))

        # check score on validation set
        with torch.no_grad():
            predicted_vals = process_batch(program, validation_input, output_type, output_size, device)
            metric, additional_params = evalfxn(predicted_vals, validation_true_vals, num_labels=num_labels)

        if use_valid_score:
            if metric < best_metric:
                best_program = copy.deepcopy(program)
                best_metric = metric
                best_additional_params = additional_params
        else:
            best_program = copy.deepcopy(program)
            best_metric = metric
            best_additional_params = additional_params

    # select model with best validation score
    program = copy.deepcopy(best_program)
    log_and_print("Validation score is: {:.4f}".format(best_metric))
    log_and_print("Average f1-score is: {:.4f}".format(1 - best_metric))
    log_and_print("Hamming accuracy is: {:.4f}".format(best_additional_params['hamming_accuracy']))
    
    return best_metric
