import numpy as np
from collections import defaultdict
import math
from itertools import repeat, chain
import numpy as np

import torch
import torch.utils.data
from torch.utils.data.sampler import BatchSampler, Sampler


def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)

# Not my code
# https://github.com/pytorch/vision/blob/master/references/detection/group_by_aspect_ratio.py
# Lot of modifications


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, group_ids, batch_size, shuffle=True, divide_batch_size_by=None, divide_batch_size_at=None):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        # print(group_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        if divide_batch_size_by is None:
            self.divide_batch_size_by = [1]
        elif not isinstance(divide_batch_size_by, list):
            self.divide_batch_size_by = list(divide_batch_size_by)
        else:
            self.divide_batch_size_by = divide_batch_size_by
        if divide_batch_size_at is None:
            self.divide_batch_size_at = [max(self.group_ids)]
        elif not isinstance(divide_batch_size_at, list):
            self.divide_batch_size_at = list(divide_batch_size_at)
        else:
            self.divide_batch_size_at = divide_batch_size_at

    def __iter__(self):
        if self.shuffle:
            self.group_ids, _ = self.sampler.shuffle()
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)
        # print(type(self.sampler))
        num_sents = 0
        for idx in self.sampler:
            # print('grp idx')
            # print(idx)
            # print(self.group_ids[idx]) # Check pour debbug
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            for i in range(len(self.divide_batch_size_at)-1, 0, -1):
                i_fix = 0  # Use for testing len(buffer)
                if len(buffer_per_group[group_id]) == self.batch_size:
                    i = 0  # Taille de batch parfaite
                if len(buffer_per_group[group_id]) >= self.batch_size/self.divide_batch_size_by[i] and group_id>=self.divide_batch_size_at[i]:
                    yield buffer_per_group[group_id]
                    # yield [data[ind] for ind in buffer_per_group[group_id]]
                    num_sents += self.batch_size/self.divide_batch_size_by[i]
                    del buffer_per_group[group_id]
                    #print('i egal : '+str(i))
                    #print('num sentences done '+str(num_sents))
                    #print('idx egal '+str(idx))
                    i_fix = i  # Use for testing len(buffer)
            assert len(buffer_per_group[group_id]) < self.batch_size/self.divide_batch_size_by[i_fix]

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_sents = len(self)
        #print('expected_num_sents')
        #print(num_sents)
        num_remaining = expected_num_sents - num_sents
        #print('num_remaining')
        #print(num_remaining)
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                for i in range(len(self.divide_batch_size_at) - 1, -1, -1):
                    if group_id > self.divide_batch_size_at[i]:
                        #print('group_id')
                        #print(group_id)
                        #samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                        #buffer_per_group[group_id].extend(samples_from_group_id[:int(remaining)])
                        assert len(buffer_per_group[group_id]) <= self.batch_size / self.divide_batch_size_by[i]
                        if len(buffer_per_group[group_id]) != 0:
                            yield buffer_per_group[group_id]
                        i = -2 # Sortir de la boucle
                        num_remaining -= len(buffer_per_group[group_id])
                        del buffer_per_group[group_id]
                        #print('ca c le remaining')
                        #print(num_remaining)
                        if num_remaining == 0:
                            break
        #print(num_remaining)
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler)



class OppositeTargetSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, target_ids, batch_size, shuffle=True, divide_batch_size_by=None, divide_batch_size_at=None):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.target_ids = target_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.list_class_1 = []
        self.list_class_0 = []
        for idx in self.sampler:
            # print(len(self.sampler))
            # print(len(self.target_ids))
            # print(self.target_ids[idx])
            if self.target_ids[idx] == 1:
                self.list_class_1.append(idx)
            else:
                self.list_class_0.append(idx)

    def __iter__(self):
        if self.shuffle:
            _, self.target_ids = self.sampler.shuffle()
            self.list_class_1 = []
            self.list_class_0 = []
            for idx in self.sampler:
                if self.target_ids[idx] == 1:
                    self.list_class_1.append(idx)
                else:
                    self.list_class_0.append(idx)
        # print(len(self.list_class_1))
        # print(len(self.list_class_0))
        for i in range(min([len(self.list_class_0), len(self.list_class_1)])):
            yield [self.list_class_0[i], self.list_class_1[i]]

    def __len__(self):
        return len(self.sampler)

class OppositeSameSizeTwoSentenceBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, group_ids, targets, batch_size=2, shuffle=True):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.group_ids, self.targets = self.sampler.shuffle()
        samples_per_targets = defaultdict(dict)
        samples_per_group = defaultdict(dict)
        # print(type(self.sampler))
        num_sents = 0
        for idx in self.sampler:
            # print('grp idx')
            # print(idx)
            # print(self.group_ids[idx]) # Check pour debbug
            group_id = self.group_ids[idx]
            target_id = self.targets[idx]
            # print(type(samples_per_group[group_id]))
            if samples_per_group[group_id] is None: # Inutile peut-être ?
                samples_per_group[group_id] = dict()
            samples_per_group[group_id][target_id] = idx
            # print(len(samples_per_group[group_id]))
            # print()
            if len(samples_per_group[group_id]) == self.batch_size:
                # yield [samples_per_group[group_id][0], samples_per_group[group_id][1]]
                # print([value for key, value in samples_per_group[group_id].items()])
                yield [value for key, value in samples_per_group[group_id].items()]
                # yield [data[ind] for ind in buffer_per_group[group_id]]
                # num_sents += self.batch_size / self.divide_batch_size_by[i]
                del samples_per_group[group_id]


    def __len__(self):
        return len(self.sampler)


class OneSentenceBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, group_ids, targets, batch_size=2, shuffle=True):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.group_ids, self.targets = self.sampler.shuffle()
        samples_per_targets = defaultdict(dict)
        samples_per_group = defaultdict(dict)
        for idx in self.sampler:
            yield [idx]


    def __len__(self):
        return len(self.sampler)
