from time import time
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

def numpy_to_lmdb(lmdb_path, data, labels, lmdb_map_size=int(50e9)):
    env = lmdb.open(lmdb_path, map_size=lmdb_map_size, map_async=True, writemap=True, create=True)
    with env.begin(write=True) as txn:
        for (i, datum) , label in zip(enumerate(data), labels):
            key = bytes('input_%s'%format(i), "ascii")
            inputs_shape = datum.shape
            outputs_shape = label.shape
            inputs = datum.flatten().tostring()
            txn.put(key, inputs)
            key = bytes('output_%s'%format(i), "ascii")
            outputs = label.flatten().tostring()
            txn.put(key, outputs)
        env.sync()
        headers = { b"input_dtype": bytes(datum.dtype.str, "ascii"),
                    b"input_shape": np.array(inputs_shape).tostring(),
                    b"output_shape": np.array(outputs_shape).tostring(),
                    b"output_dtype": bytes(label.dtype.str, "ascii"),
                    b"output_name": bytes('output_', "ascii"),
                    b"input_name": bytes('input_', "ascii")}
        for key, val in headers.items():
            txn.put(key, val)
        txn.put(b"header_entries", bytes(len(list(headers.items()))))
        env.sync()


class ABFDataSet(Dataset):
    """ ABF data set on lmdb."""
    def __init__(self, lmdb_path, key_base = 'sample', input_transform=None, target_transform=None,
                                        debug=True):
        self.debug = debug
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(self.lmdb_path, create=False, readahead=False, readonly=True, writemap=False, lock=False)
        self.num_samples = (self.env.stat()['entries'] - 6)//2 ## TODO: remove hard-coded # of headers by storing #samples key, val
        self.first_record = 0
        self.records = np.arange(self.first_record, self.num_samples)
        with self.env.begin(write=False) as txn:
            input_shape = np.frombuffer(txn.get(b"input_shape"), dtype='int64')
            output_shape = np.frombuffer(txn.get(b"output_shape"), dtype='int64')
            input_dtype = np.dtype(txn.get(b"input_dtype").decode("ascii"))
            output_dtype = np.dtype(txn.get(b"output_dtype").decode("ascii"))
            output_name = txn.get(b"output_name").decode("ascii")
            input_name = txn.get(b"input_name").decode("ascii")
        self.data_specs={'input_shape': list(input_shape), 'target_shape': list(output_shape), 
            'target_dtype':output_dtype, 'input_dtype': input_dtype, 'target_key':output_name, 'input_key': input_name}
        self.input_keys = [bytes(self.data_specs['input_key']+str(idx), "ascii") for idx in self.records]
        self.target_keys = [bytes(self.data_specs['target_key']+str(idx), "ascii") for idx in self.records]
        self.print_debug("Opened lmdb file %s, with %d samples" %(self.lmdb_path, self.num_samples))
        self.input_transform = input_transform
        self.target_transform = target_transform

    def print_debug(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def __len__(self):
        ## TODO: Need to specify how many records are for headers
        return self.num_samples

    def __getitem__(self, idx):
        # outside_func(idx)
        input_key = self.input_keys[idx]
        target_key = self.target_keys[idx]
        with self.env.begin(write=False, buffers=True) as txn:
            input_bytes = txn.get(input_key)
            target_bytes = txn.get(target_key)
        inputs = np.frombuffer(input_bytes, dtype=self.data_specs['input_dtype'])
        inputs = inputs.reshape(self.data_specs['input_shape'])
        targets = np.frombuffer(target_bytes, dtype=self.data_specs['target_dtype'])
        targets = targets.reshape(self.data_specs['target_shape'])
        self.print_debug('read inputs # %d with size %d' %(idx, inputs.size))
        if self.input_transform is not None:
            inputs = self.transform_input(inputs)
        if self.target_transform is not None:
            targets = self.transform_target(targets)
        return {'input':torch.from_numpy(inputs), 'target':torch.from_numpy(targets)}

    @staticmethod
    def transform_target(targets):
        pass

    @staticmethod
    def transform_input(inputs):
        ## TODO: implement addition of poisson noise, global affine distortions, and crop.
        # The above transformations, in sequence, are the only ones that should be used.
        pass

    def __repr__(self):
        pass


class ABFDataSetMulti(Dataset):
    """ ABF data set on lmdb."""
    def __init__(self, lmdb_dir, key_base = 'sample', input_transform=None, target_transform=None,
                                        input_shape=(1,85,120), target_shape=(3,),
                                        debug=True):
        self.debug = debug
        self.lmdb_path = [os.path.join(lmdb_dir, lmdb_path) for lmdb_path in os.listdir(lmdb_dir)]
        self.db = [lmdb.open(lmdb_path, readahead=False, readonly=True, writemap=False, lock=False)
                            for lmdb_path in self.lmdb_path]

        ## TODO: Need to specify how many records are for headers at __init__: now 2
        self.db_records = [db.stat()['entries'] - 2 for db in self.db]
        self.db_idx_sum = np.cumsum(self.db_records)

        with self.db[0].begin(write=False) as txn:
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
        ## TODO: Need to specify how many records are for headers at __init__: now 2
        return sum(self.db_records)


    def __getitem__(self, idx):
        # outside_func(idx)
        # map record index to lmdb file index
        db_idx = np.argmax(idx < self.db_idx_sum)
        if db_idx > 0: 
            idx -= np.sum(self.db_records[:db_idx])
        assert idx >= 0 and idx < sum(self.db_records) , print(idx, sum(self.db_records))
        # fetch records
        # try:
        with self.db[db_idx].begin(write=False, buffers=True) as txn:
            key = bytes('%s_%i' %(self.key_base, idx), "ascii")
            self.print_debug('going into lmdb file %s, reading %d' % (self.lmdb_path[db_idx], idx )) 
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
        # except AttributeError:
            # print("key: %s in file: %s" %(key, self.lmdb_path[db_idx]))

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

def benchmark_io(lmdb_path, mpi_rank, step=100, warmup=100, max_batches=1000,
                batch_size=512, shuffle=True, num_workers=20, pin_mem=True,
                gpu_copy=True, debug=False):
    """ Measure I/O performance of lmdb and multiple python processors during
        training.
    """
    abfData = ABFDataSet(lmdb_path, debug=debug)
    data_loader = DataLoader(abfData, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_mem)
    bandwidths=[]
    t = time()
    with torch.cuda.device(mpi_rank):
        for batch_num, batch in enumerate(data_loader):
            if gpu_copy:
                batch['input'].cuda(non_blocking=pin_mem)
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

def benchmark_io_multi(lmdb_path, mpi_rank, step=100, warmup=100, max_batches=1000,
                batch_size=512, shuffle=True, num_workers=20, pin_mem=True,
                gpu_copy=True, debug=False):
    """ Measure I/O performance of lmdb and multiple python processors during
        training.
    """
    abfData = ABFDataSetMulti(lmdb_path, debug=debug)
    data_loader = DataLoader(abfData, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_mem)
    bandwidths=[]
    t = time()
    with torch.cuda.device(mpi_rank):
        for batch_num, batch in enumerate(data_loader):
            if gpu_copy:
                batch['input'].cuda(non_blocking=pin_mem)
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
