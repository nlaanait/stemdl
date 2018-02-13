#!/bin/bash
gpus=2
script=stemdl_regress_horovod_json_input.py
echo mpirun --allow-run-as-root -np $gpus python $script
mpirun --allow-run-as-root -np $gpus python $script
rm stemdl_checkpoint/ -r
