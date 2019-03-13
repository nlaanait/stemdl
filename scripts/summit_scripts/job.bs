#!/bin/bash -l
#BSUB -P LRN001 
##BSUB -J atomai_take2 
#BSUB -J @not@specified@ 
#BSUB -o logs.o%J
#BSUB -W 00:15
#BSUB -nnodes 256 
#BSUB -alloc_flags "smt4 nvme maximizegpfs" 
#BSUB -q batch
##BSUB -N
##BSUB -csm y
##BSUB -alloc_flags "smt4 gpumps maximizegpfs nvme"

extract_json_field(){
  grep $1 input_flags.json | awk -v FS="(value\":|,)" '{print $3}'
}

NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
BUILDS=${PROJWORK}/lrn001/nl/builds_test
BUILDS=${PROJWORK}/lrn001/jqyin/native-build/latest

### modules ###
module load cuda/9.2.148
module load spectrum-mpi
module unload darshan-runtime
module unload xalt 

### TF ###
export TF_CPP_MIN_LOG_LEVEL="2"
export TF_CUDNN_USE_AUTOTUNE=1 #1
export TF_AUTOTUNE_THRESHOLD=2 #2
export TF_ENABLE_WINOGRAD_NONFUSED=1
export TF_ENABLE_XLA=0
#export TF_GPU_THREAD_COUNT=2
export TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT=1
#export TF_GPU_THREAD_MODE="gpu-private"
####################


### python ###
export LD_LIBRARY_PATH=$BUILDS/nccl/lib:$BUILDS/cudnn/lib:${CUDA_DIR}/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PATH=$BUILDS/anaconda3/bin:$PATH
export PYTHONIOENCODING="utf8"
#############

### nvme staging ###
DATA="${PROJWORK}/lrn001/nl/sims/data/lmdb_test"
FILETYPE="lmdb"
#DATA="${PROJWORK}/lrn001/nl/sims/data/tfrecords"
#FILETYPE="tfrecord"
LOG="printout_nvme.log"
NVME_PURGE=0
CMD="python -u nvme_stage.py $DATA $FILETYPE"
jsrun -n${NODES} -a 6 -c 42 -r 1 $CMD > $LOG 
DATA="/mnt/bb/${USER}"
jswait 1
####################

### stemdl ###
LOG="output_logs/printout_${NODES}_${LSB_JOBID}_group_perlayer.log"
SCRIPT="stemdl_run.py"
NROOT="FCDenseNet_custom" 
HYPER="hyper_params.json"
INPUT="input_flags.json"
NETWORK="network_${NROOT}.json"
export CKPT_DIR="checkpoints"
CKPT="/mnt/bb/${USER}/${NROOT}_${NODES}_${LSB_JOBID}_checkpoint"
cpus=1
ILR=1.e-3
BN_DECAY=0.1
EPOCH_PER_DECAY=5.0
EPOCH_PER_SAVE=200.0
EPOCH_PER_VALIDATE=200.0
SCALING=1.0
FP=fp16
MODE="train"
MAX_STEPS=10000
BATCH=1
CMD="python -u ${SCRIPT}   --hvd_fp16  --nvme --filetype ${FILETYPE} --data_dir ${DATA}  --${FP}  --cpu_threads $cpus --mode $MODE --validate_epochs $EPOCH_PER_VALIDATE --save_epochs $EPOCH_PER_SAVE --batch_size $BATCH  --log_frequency 50 --max_steps $MAX_STEPS --network_config ${NETWORK} --checkpt_dir ${CKPT} --ilr ${ILR} --bn_decay ${BN_DECAY} --scaling ${SCALING} --input_flags ${INPUT} --hyper_params ${HYPER}"
#################

### HOROVOD ####
#export HOROVOD_HIERARCHICAL_ALLREDUCE=1
#export HOROVOD_GPU_ALLGATHER="MPI"
#export HOROVOD_MPI_THREADS_DISABLE=1
export HOROVOD_HIERARCHICAL_ALLGATHER=0
export HOROVOD_HIERARCHICAL_ALLREDUCE=0
#export HOROVOD_TIMELINE="timeline.${LSB_JOBID}"
export HOROVOD_GROUPED_ALLREDUCES=1
export HOROVOD_CYCLE_TIME=2
#export HOROVOD_FUSION_THRESHOLD=67108864
###############

### mpi/cudnn tracing ###
#export OMPI_LD_PRELOAD_POSTPEND=${OLCF_SPECTRUM_MPI_ROOT}/lib/libmpitrace.so
#export MPI_ASC_OUTPUT="Y"
#export CUDNN_LOGINFO_DBG=0
#export CUDNN_LOGDEST_DBG="output_logs/cudnn_${NODES}_${LSB_JOBID}.log"
################

### pami ibv ###
#export PAMI_IBV_ENABLE_DCT=1
#export PAMI_ENABLE_STRIPING=1
#export PAMI_IBV_ADAPTER_AFFINITY=1
#export PAMI_IBV_QP_SERVICE_LEVEL=8
#export PAMI_IBV_ENABLE_OOO_AR=1
#export PAMI_IBV_DEVICE_NAME="mlx5_0:1,mlx5_3:1";
#export PAMI_IBV_ADAPTER_AFFINITY=1
#export PAMI_IBV_ENABLE_OOO_AR=0
#export PAMI_ENABLE_STRIPING=1
#export PAMI_IBV_DEVICE_NAME="mlx5_0:1,mlx5_3:1"
#export PAMI_IBV_DEVICE_NAME_1="mlx5_3:1"
###############

### nccl ###
#export NCCL_MIN_NRINGS=4
#export NCCL_IB_CUDA_SUPPORT=1
#export NCCL_MAX_NRINGS=16
#export NCCL_IB_HCA="mlx5_0:1,mlx5_3:1,^mlx5_0:2,^mlx5_1:2"
#export NCCL_BUFFSIZE=16777216 
#export NCCL_LL_THRESHOLD=1048576 
#LOG="printout_${NODES}_${NCCL_LL_THRESHOLD}_${NCCL_BUFFSIZE}.log"
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_FILE="nccl_debug_%h_%p.log"
###################

#jsrun -n${NODES} -a 6 -c 42 -g 6 -r 1 --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./launch.sh "${CMD}" > $LOG
jsrun -n${NODES} -a 6 -c 42  -g 6 -r 1 --bind=proportional-packed:7 --launch_distribution=packed ${CMD} > $LOG
