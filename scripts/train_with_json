#!/bin/bash
gpus=2
batch=64
logfile=batch-$batch-gpus-$gpus
script=stemdl_regress_horovod_json_input.py
echo mpirun --allow-run-as-root -np $gpus python $script &> $logfile
mpirun --allow-run-as-root -np $gpus python $script &>> $logfile
rm stemdl_checkpoint/ -r