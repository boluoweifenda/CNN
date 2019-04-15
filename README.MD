# TensorFlow-Slimmer image classification model library

A simple and extensible project based on 
[TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)
for 
- fast demonstrations of various ideas
- an optimal toolbox and framework for CNN beginners
- exercises for Tsinghua lecture: Neuromorphic Computing Theory and System


## Prerequisites
- A computer with NVIDIA GPUs
- CUDA + CuDNN
- Tensorflow (GPU version)
- python3
- tqdm


## Preparing datasets
Download and generate mnist, fashion and cifar datasets by:
```bash
python3 data/xxx/download_and_convert_xxx.py
```
For imagenet dataset, please follow the instructions in 
[an automated script](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data),
and change the destinations in `/source/datasets/dataset_factory` to the converted tf.record file locations


## Configuring training hyperparameters

Change or new an option file in `source/options/`, there are some templates for beginners.


### Training devices
Change your configurations in the file

- Single GPU trainig with GPU-2: `gpu_list = [2]`
- Multi-GPU training with GPU-1 and GPU-3: `gpu_list = [1,3]`
- CPU-only training (deprecated): `gpu_list = []` 


### Network models

Change or new a model file in `source/nets/`, and register it in `nets_factory.py`

You can also modify and add other components, such as preprocessing, dataset. 
Don't forget to register them in individual factory files.

## Start training:
In case the option file name in `/source/options/` is `your_training_options.py` 
```bash
cd source/
python3 top.py -o your_training_options
```
The training log files will be printed and saved in `log/time_your_options_title.txt`

You can check the training details in log files, and derive statistics for drawing curves.

And the model will be saved in `model/time_your_options_title` if you have configured the saving parameters.
a

## Exercise for Tsinghua lecture: Neuromorphic Computing Theory and System

