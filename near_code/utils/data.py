import random
import torch
import numpy as np
from collections.abc import Iterable


def flatten_batch(batch):
    if not isinstance(batch[0], Iterable) or len(batch[0]) == 1:
        return batch
    new_batch = []
    for traj_list in batch:
        new_batch.extend(traj_list)
    return new_batch

def flatten_tensor(batch_out):
    return torch.cat(batch_out)

def pad_minibatch(minibatch, num_features=-1, pad_token=-1, return_max=False):
    #minibatch to be of dimension [num_sequences, num_features, len of each sequence (variable)]
    #adapted from Will Falcon's blog post
    batch_lengths = [len(sequence) for sequence in minibatch]
    batch_size = len(minibatch)
    longest_seq = max(batch_lengths)
    padded_minibatch = torch.ones((batch_size, longest_seq, num_features)) * pad_token
    for i, seq_len in enumerate(batch_lengths):
        seq = minibatch[i]
        if num_features == 1:
            seq = seq.unsqueeze(1)
        padded_minibatch[i, 0:seq_len] = seq[:seq_len]
    if return_max:
        return padded_minibatch, batch_lengths, longest_seq
    else:
        return padded_minibatch, batch_lengths

def unpad_minibatch(minibatch, lengths, listtoatom=False):
    new_minibatch = []
    for idx, length in enumerate(lengths):
        if listtoatom:
            new_minibatch.append(minibatch[idx][length-1])
        else:
            new_minibatch.append(minibatch[idx][:length])
    return new_minibatch

def dataset_tolists(trajs, labels):
    assert len(trajs) == len(labels)

    dataset = []
    for k, traj in enumerate(trajs):
        traj_list = []
        for t in range(len(traj)):
            traj_list.append(traj[t])

        label = torch.tensor(labels[k]).long()
        dataset.append([traj_list, label])

    return dataset

def normalize_data(train_data, valid_data, test_data):
    """Normalize features wrt. mean and std of training data."""
    _, seq_len, input_dim = train_data.shape
    train_data_reshape = np.reshape(train_data, (-1, input_dim))
    test_data_reshape = np.reshape(test_data, (-1, input_dim))
    features_mean = np.mean(train_data_reshape, axis=0)
    features_std = np.std(train_data_reshape, axis=0)
    train_data_reshape = (train_data_reshape - features_mean) / features_std
    test_data_reshape = (test_data_reshape - features_mean) / features_std
    train_data = np.reshape(train_data_reshape, (-1, seq_len, input_dim))
    test_data = np.reshape(test_data_reshape, (-1, seq_len, input_dim))
    if valid_data is not None:
        valid_data_reshape = np.reshape(valid_data, (-1, input_dim))
        valid_data_reshape = (valid_data_reshape - features_mean) / features_std
        valid_data = np.reshape(valid_data_reshape, (-1, seq_len, input_dim))
    return train_data, valid_data, test_data

def create_minibatches(all_items, batch_size):
    num_items = len(all_items)
    batches = []
    def create_single_minibatch(idxseq):
        curr_batch = []
        for idx in idxseq:
            curr_batch.append((all_items[idx]))
        return curr_batch
    item_idxs = list(range(num_items))
    while len(item_idxs) > 0:
        if len(item_idxs) <= batch_size:
            batch = create_single_minibatch(item_idxs)
            batches.append(batch)
            item_idxs = []
        else:
            # get batch indices
            batchidxs = []
            while len(batchidxs) < batch_size:
                rando = random.randrange(len(item_idxs))
                index = item_idxs.pop(rando)
                batchidxs.append(index)
            batch = create_single_minibatch(batchidxs)
            batches.append(batch)
    return batches

def prepare_datasets(train_data, valid_data, test_data, train_labels, valid_labels, test_labels, normalize=True, train_valid_split=0.7, batch_size=32):
    if normalize:
        train_data, valid_data, test_data = normalize_data(train_data, valid_data, test_data)

    trainset = dataset_tolists(train_data, train_labels) 
    testset = dataset_tolists(test_data, test_labels)

    if valid_data is not None and valid_labels is not None:
        validset = dataset_tolists(valid_data, valid_labels)
    # Split training for validation set if validation set is not provided.
    elif train_valid_split < 1.0:
        split = int(train_valid_split*len(train_data))
        validset = trainset[split:]
        trainset = trainset[:split]
    else:
        split = int(train_valid_split)
        validset = trainset[split:]
        trainset = trainset[:split]

    # Create minibatches for training
    batched_trainset = create_minibatches(trainset, batch_size)

    return batched_trainset, validset, testset
