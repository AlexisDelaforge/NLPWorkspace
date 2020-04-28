import numpy as np
from collections import defaultdict
import math
from itertools import repeat, chain

import torch
import torch.utils.data
from torch.utils.data.sampler import BatchSampler, Sampler


def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)

# Not my code
# https://github.com/pytorch/vision/blob/master/references/detection/group_by_aspect_ratio.py
# No modification


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
    def __init__(self, sampler, group_ids, batch_size, divide_batch_size_by=None, divide_batch_size_at=None):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size
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
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            # print('grp idx')
            # print(idx)
            # print(self.group_ids[idx]) # Check pour debbug
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            for i in range(len(self.divide_batch_size_at)-1, -1, -1):
                if len(buffer_per_group[group_id]) == self.batch_size or (len(buffer_per_group[group_id]) >= self.batch_size/self.divide_batch_size_by[i] and group_id>self.divide_batch_size_at[i]):
                    yield buffer_per_group[group_id]
                    # yield [data[ind] for ind in buffer_per_group[group_id]]
                    num_batches += 1
                    del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size