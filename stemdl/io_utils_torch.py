from time import time
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

def numpy_to_lmdb(lmdb_path, data, labels, lmdb_map_size=int(1e12)):
    env = lmdb.open(lmdb_path, map_size=lmdb_map_size)
    with env.begin(write=True) as txn:
        for (i, datum) , label in zip(enumerate(data), labels):
            key = bytes('sample_%s'%format(i), "ascii")
            sample = np.concatenate((datum.flatten(), label.flatten().astype(np.float16)))
            sample = sample.tostring()
            txn.put(key, sample)
        headers = {b"data_dtype": bytes(data.dtype.name, "ascii"),
                   b"data_shape": np.array(data.shape).tostring()}
        for key, val in headers.items():
            txn.put(key, val)


class ABFDataSet(Dataset):
    """ ABF data set on lmdb."""
    def __init__(self, lmdb_path, key_base = 'sample', input_transform=None, target_transform=None,
                                        input_shape=(1,85,120), target_shape=(3,),
                                        debug=True):
        self.debug = debug
        self.lmdb_path = lmdb_path
        self.db = lmdb.open(self.lmdb_path, readahead=False,
        readonly=True, writemap=False, lock=False)
        with self.db.begin(write=False) as txn:
            self.dtype = np.dtype(txn.get(b"data_dtype"))
        self.print_debug("read dtype %s from lmdb file %s" %(format(self.dtype),
                                                            self.lmdb_path))
        #TODO: add shapes to lmdb headers.
        #TODO: add dtypes to lmbd headers.
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.key_base = key_base
        self.input_transform = input_transform
        self.target_transform = target_transform

    def print_debug(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def __len__(self):
        ## TODO: Need to specify how many records are for headers
        return self.db.stat()['entries'] - 2

    def __getitem__(self, idx):
        # outside_func(idx)
        with self.db.begin(write=False, buffers=True) as txn:
            key = bytes('%s_%i' %(self.key_base, idx), "ascii")
            bytes_buff = txn.get(key)
            sample = np.frombuffer(bytes_buff, dtype=self.dtype)
        input_size = np.prod(np.array(self.input_shape))
        target_size = np.prod(np.array(self.target_shape))
        input = sample[:input_size].astype('float32')
        target = sample[-target_size:].astype('float64')
        self.print_debug('read input %d with size %d' %(idx, input.size))
        if self.input_transform is not None:
            input = self.transform_input(input)
        if self.target_transform is not None:
            target = self.transform_target(target)

        input = input.reshape(self.input_shape)
        target = target.reshape(self.target_shape)

        return {'input':torch.from_numpy(input), 'target':torch.from_numpy(target)}

    @staticmethod
    def transform_target(target):
        if target.dtype != 'float64':
            return target.astype('float64')

    @staticmethod
    def transform_input(input):
        if input.dtype != 'float32':
            return input.astype('float32')

    def __repr__(self):
        pass

def set_io_affinity(mpi_rank, mpi_size, debug=True):
    """
    Set the affinity based on available cpus, mpi (local) rank, and mpi (local)
    size. Assumes mpirun binding is none.
    """
    if debug: 
        print("Initial Affinity %s" % os.sched_getaffinity(0))
    total_procs = len(os.sched_getaffinity(0))
    max_procs = total_procs // mpi_size
    new_affnty = range( mpi_rank * max_procs, (mpi_rank + 1) * max_procs)
    os.sched_setaffinity(0, new_affnty)
    if debug:
        print("New Affinity %s" % os.sched_getaffinity(0))
    return new_affnty

def benchmark_io(lmdb_path, step=100, warmup=100, max_batches=1000,
                batch_size=512, shuffle=True, num_workers=20, pin_mem=True,
                gpu_copy=True):
    """ Measure I/O performance of lmdb and multiple python processors during
        training.
    """
    abfData = ABFDataSet(lmdb_path, debug=False)
    data_loader = DataLoader(abfData, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_mem)
    bandwidths=[]
    t = time()
    cuda = torch.device('cuda')
    for batch_num, batch in enumerate(data_loader):
        if gpu_copy:
            batch['input'].cuda(non_blocking=True)
        if batch_num % step == 0:
            print('loaded batch %d' % batch_num)
            print('input:', batch['input'].size(), 'target:', batch['target'].size())
            t_run = time() - t
            size = torch.prod(torch.tensor(batch['input'].size())).numpy() * \
            batch['input'].element_size() * step
            t = time()
            if batch_num > warmup:
                bandwidths.append(size/1024e6/t_run)
                print('Bandwidth: %2.3f (GB/s)' % (size/1024e6/t_run))
            t = time()
        if batch_num == max_batches:
            break
    bandwidths = np.array(bandwidths)
    print('Total Bandwidth: %2.2f +/- %2.2f (GB/s)' %(bandwidths.mean(), bandwidths.std()))
    return bandwidths.mean(), bandwidths.std()
