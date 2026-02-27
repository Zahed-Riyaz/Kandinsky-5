import math
from collections import defaultdict
from itertools import chain, repeat
from torch.utils.data import BatchSampler


def repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)


class GroupedBatchSampler(BatchSampler):
    def __init__(self, sampler, group_indices, batch_size):
        self.sampler = sampler
        self.group_indices = group_indices
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_idx = self.group_indices[idx]
            buffer_per_group[group_idx].append(idx)
            samples_per_group[group_idx].append(idx)
            if len(buffer_per_group[group_idx]) == self.batch_size:
                yield buffer_per_group[group_idx]
                num_batches += 1
                del buffer_per_group[group_idx]
            assert len(buffer_per_group[group_idx]) < self.batch_size

        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            for group_idx, _ in sorted(buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_idx])
                samples_from_group_idx = repeat_to_at_least(samples_per_group[group_idx], remaining)
                buffer_per_group[group_idx].extend(samples_from_group_idx[:remaining])
                assert len(buffer_per_group[group_idx]) == self.batch_size
                yield buffer_per_group[group_idx]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size

    def set_epoch(self, epoch: int):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class LoopWrapper:
    def __init__(self, dataloader, sampler=None):
        self.dataloader = dataloader
        self.sampler = sampler
        self.epoch = 0

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self.loop_forever()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def set_epoch(self, epoch: int):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def loop_forever(self):
        self.set_epoch(self.epoch)
        iterator = iter(self.dataloader)
        while True:
            try:
                data = next(iterator)
                yield data
            except StopIteration:
                if self.sampler is not None:
                    self.epoch += 1
                    self.set_epoch(self.epoch)
                iterator = iter(self.dataloader)