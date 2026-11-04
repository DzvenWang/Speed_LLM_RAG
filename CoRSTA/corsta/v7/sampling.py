"""Chunk-aware TRAIN shuffle for mmap replay without fully sequential epochs.

Randomize chunk order and within-chunk indices, then pack adjacent shuffled
chunks into batches. All samples are visited exactly once per epoch.
"""
from __future__ import annotations
import math
import torch
from torch.utils.data import Sampler,DataLoader

class ChunkShuffleBatchSampler(Sampler):
    def __init__(self,num_samples,chunk_size,batch_size):
        self.n=int(num_samples);self.chunk=int(chunk_size);self.batch=int(batch_size)
        if min(self.n,self.chunk,self.batch)<1:raise ValueError('Sampler size must be positive')
    def __iter__(self):
        chunks=torch.randperm(math.ceil(self.n/self.chunk)).tolist()
        # Pack chunks in random order; preserving within-batch local chunk reads.
        group=[]
        for c in chunks:
            ids=list(range(c*self.chunk,min(self.n,(c+1)*self.chunk)))
            perm=torch.randperm(len(ids)).tolist()
            ids=[ids[i] for i in perm]
            for idx in ids:
                group.append(idx)
                if len(group)>=self.batch:
                    yield group
                    group=[]
        if group:yield group
    def __len__(self):return math.ceil(self.n/self.batch)

def replay_train_loader(dataset,batch_size,workers):
    return DataLoader(dataset,batch_sampler=ChunkShuffleBatchSampler(len(dataset),dataset.chunk,batch_size),
       num_workers=workers,pin_memory=torch.cuda.is_available(),
       persistent_workers=(workers>0),prefetch_factor=2 if workers>0 else None)
