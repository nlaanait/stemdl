# `STEMDL`
A Python package for distributed deep learning with a special focus on inverse problems in materials imaging.  
`stemdl` was used in the following (applied and fundamental) deep learning research projects:   
1. *3-D reconstruction of Structural Distortions from Electron Microscopy* ([Link to Paper](https://arxiv.org/abs/1902.06876))  
2. *27,600 V100 GPUs and 7MW(h) of Power to solve an age-old scientific inverse problem* ([Link to Paper](https://arxiv.org/abs/1909.11150) and [Medium story](https://medium.com/syncedreview/nvidia-ornl-researchers-train-ai-model-on-worlds-top-supercomputer-using-27-600-nvidia-gpus-1165e0d5da7b) )
3. *YNet: a Physics-Constrainted and Semi-Supervised Learning Approach to Inverse Problems*
---
#### Getting Started
See __scripts__ folder for the following:  
1. __stemdl_run.py__:    
  Python script. Runs from the CLI to setup Neural Nets and start training/evaluation operations.
2. __generate_json.py__:  
  Python script. Generates .json files needed as input for stemdl_run.py
---
#### Brief description of Modules:  
1. __inputs.py__:  
  Classes to read training/evaluation data, create training batches, and image transformations.  
  Can handle I/O ops on TFRecords, numpy arrays, and lmdb files
2. __network.py__:  
  Classes to setup various kinds of Neural Nets (ConvNets, ResNets, etc...)  
3. __runtime.py__:  
  Functions and Classes to perform (low-level) network training/evaluation
4. __io_utils.py__:  
  Functions to generate .json files for model architectures input files, hyperparameters, and training runs configurations
5. __losses.py__:   
   Functions to generate and manipulate loss functions
6. __optimizers.py__:   
   Optimizer setup and gradients pre-processing and reduction
7. __automatic_loss_scaler.py__:  
   Python module for dynamic loss scaling during fp16 training (taken as is from [OpenSeq2Seq](https://nvidia.github.io/OpenSeq2Seq/html/index.html))
---
#### Software Requirements:  
1. __numpy__ >= 1.13  
2. __tensorflow__ >=1.2
3. __python__ 3.6
4. __horovod__ >=0.16

#### Hardware Requirements:  
1. CUDA compatible GPU >=1


 


