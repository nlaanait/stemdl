import os, subprocess, shlex, sys
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

def nvme_staging(data_dir, data_tally, eval_data=False):
    user = os.environ.get('USER')
    nvme_dir = '/mnt/bb/%s' %(user)
    index = comm_rank
    if not eval_data:
        # training data
        src = os.path.join(data_dir, 'batch_train_%d.db' % comm_rank)
        if not os.path.exists(src):
            src = u'%s/batch_train_0.db' % data_dir
        src = check_file(data_tally, src, mode="train")
        trg = '%s/batch_train_%d.db' %(nvme_dir, comm_rank)
        cp_args = "cp -r %s %s" %(src, trg)
        cp_args = shlex.split(cp_args)
        if not os.path.exists(trg):
            try:
                subprocess.run(cp_args, check=True, timeout=120)
                print("rank %d:staged %s" % (comm_rank, trg ))
            except subprocess.SubprocessError as e:
                print("rank %d: %s" % (comm_rank, format(e)))
    # evaluation data
        src = "%s/batch_eval_%d.db" %(data_dir, index)
        if not os.path.exists(src):
            src =  "%s/batch_eval_0.db" % data_dir
        src = check_file(data_tally, src, mode="eval")
        trg = "%s/batch_eval_%d.db" %(nvme_dir, comm_rank)
        cp_args = "cp -r %s %s" %(src, trg)
        cp_args = shlex.split(cp_args)
        if not os.path.exists(trg):
            try:
                subprocess.run(cp_args, check=True, timeout=120)
                print("rank %d:staged %s" % (comm_rank, trg ))
            except subprocess.SubprocessError as e:
                print("rank %d: %s" % (comm_rank, format(e)))
    else:
        src = "%s/batch_eval_%d.db" %(data_dir, index)
        if not os.path.exists(src):
            src =  "%s/batch_eval_0.db" % data_dir
        src = check_file(data_tally, src, mode="eval")
        trg = "%s/batch_eval_%d.db" %(nvme_dir, comm_rank)
        cp_args = "cp -r %s %s" %(src, trg)
        cp_args = shlex.split(cp_args)
        if not os.path.exists(trg):
            try:
                subprocess.run(cp_args, check=True, timeout=120)
                print("rank %d:staged %s" % (comm_rank, trg ))
            except subprocess.SubprocessError as e:
                print("rank %d: %s" % (comm_rank, format(e)))

def check_file(tally_path, src, mode="train"):
    tally_arr = np.load(tally_path)
    mask = np.array([itm.find('_%s_' % mode) for itm in tally_arr['filepath']])
    mask[mask >= 0] = 1
    mask[mask < 0] = 0
    mask = mask.astype(np.bool)
    tally_arr = tally_arr[mask]
    cnt = tally_arr['num_samples'][np.where(tally_arr['filepath'] == src)[0]]
    if cnt <= 0 :
        idx = np.where(tally_arr['num_samples'] > 4)[0]
        rand = np.random.randint(0, idx.size)
        new_src = tally_arr['filepath'][idx[rand]]
        print("swapping %s with %s" %(src, new_src))
        return new_src
    return src
    

def nvme_staging_ftf(data_dir):
    user = os.environ.get('USER')
    nvme_dir = '/mnt/bb/%s' %(user)
    src = "%s/batch_%d.tfrecords" %(data_dir, comm_rank)
    trg = "%s/batch_%d.tfrecords" %(nvme_dir, comm_rank)
    #cp_args = "cp -r %s/batch_%d.db %s/batch_%d.db" %(data_dir, comm_rank, nvme_dir, comm_rank)
    cp_args = "cp %s %s" %(src, trg)
    cp_args = shlex.split(cp_args)
    if not os.path.exists(trg):
        try:
            subprocess.run(cp_args, check=True)
        except subprocess.SubprocessError as e:
            print("rank %d: %s" % (comm_rank, format(e)))

def nvme_purging():
    user = os.environ.get('USER')
    nvme_dir = '/mnt/bb/%s' %(user)
    cp_args = "rm -r %s/batch_%d.db " % (nvme_dir, comm_rank)
    cp_args = shlex.split(cp_args)
    try:
        subprocess.run(cp_args, check=True)
    except subprocess.SubprocessError as e:
        print("rank %d: %s" % (comm_rank, format(e)))

if __name__ == "__main__":
    user = os.environ.get('USER')
    mpi_host = MPI.Get_processor_name()
    nvme_dir = '/mnt/bb/%s' % user
    if len(sys.argv) > 3:
        data_dir = sys.argv[-4]
        file_type = sys.argv[-2]
        data_tally = sys.argv[-3]
        mode = sys.argv[-1]
        eval_data = True if mode == 'eval' else False 
        if not os.path.exists(data_tally):
            print('data tally file path does not exists, exiting...')
            sys.exit(1)
        if file_type == 'tfrecord':
            nvme_staging = nvme_staging_ftf
        local_rank_0 = not bool(comm_rank % 6)
        if local_rank_0:
            print('nvme contents on %s: %s '%(mpi_host,format(os.listdir(nvme_dir))))
        # stage
        if local_rank_0 : print('begin staging on %s' % mpi_host)
        nvme_staging(data_dir, data_tally, eval_data=eval_data)
        if local_rank_0:
            print('all local ranks finished nvme staging on %s' % mpi_host)
        # check stage 
        if local_rank_0:
           print('nvme contents on %s: %s '%(mpi_host, format(os.listdir(nvme_dir))))
        comm.Barrier()
        sys.exit(0)
    else:
        print('Need paths to data and tally array, and file type')
