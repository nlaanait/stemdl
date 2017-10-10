# STEMDL
A python package for deep learning-based analysis of Scanning Transmission Electron Microscopy. 
Longer description coming one of those days...     
To get started see __scripts__ folder for the following:  
1. __stemdl_run.py__:    
  Python script. Runs from the CLI to setup Neural Nets and start training/evaluation operations.
2. __generate_json.py__:  
  Python script. Generates .json files needed as input for stemdl_run.py

Here's a brief description of current modules:  
1. __inputs.py__:  
  Classes to read training/evaluation data, create training batches, and image transformations.  
  Can handle I/O ops on TFRecords and numpy arrays.
2. __network.py__:  
  Classes to setup various kinds of Neural Nets (ConvNets, ResNets, etc...)  
3. __runtime.py__:  
  Functions that perform network training/evaluation, generation of Tensorboard summaries,  also sets FLAGS that describe data, saving, training params.  
4. __io_utils.py__:  
  Functions to generate .json files for Neural Nets architecture input files and their hyper-parameters.    
  
#### Software Requirements:  
1. __numpy__ >= 1.13.  
2. __tensorflow__ >=1.2.
3. __python__ 3

#### Hardware Requirements:  
1. CUDA compatible GPU >=1.


 


