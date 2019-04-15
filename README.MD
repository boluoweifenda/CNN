# TensorFlow-Slimmer image classification model library

A simple and extensible project based on 
[TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)
for fast demonstrations of convolutional neural network training



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

Change or new an option file in `/options`, there are some templates for beginners.


### Training devices
Change your configurations in the file

Single GPU trainig with GPU-2 `gpu_list = [2]`



Change your configurations in the file
```bash
gedit source/opt.py
```

## Config network models
Change your configurations in the file
```bash
gedit source/nets/mlp.py
```
or
```bash
gedit source/nets/lenet5.py
```
or add your own network files, don't forget to register them in nets_factory.py

You can also adjust and add other components, such as preprocessing, dataset, and register them in individual factory files.

## Train
Start training:
```bash
cd source/
python top.py
```
The training log files will be saved in log/xxxx.txt

You can check the training details in log files, and derive statistics for drawing curves.

And the model will be saved in model/xxxx if you have configured in opt.py
a

## Exercise for Tsinghua lecture: Neuromorphic Computing Theory and System

