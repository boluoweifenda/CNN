# A simplified project based on TensorFlow-Slim image classification model library

Code example for convolutional neural network training

## Prerequisites
- A computer with NVIDIA GPUs
- CUDA + CuDNN
- Tensorflow (GPU version)
- python3
- tqdm


## Data
Download and generate mnist, fashion and cifar datasets by:
```bash
python3 data/xxx/download_and_convert_xxx.py
```

## Config training devices
Change your configurations in the file
```bash
gedit source/opt.py
gpu_list = [0]
```
to use GPU0
or
```bash
gpu_list = []
```
to use cpu


## Config training hyperparameters
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


## Exercise for Tsinghua lecture: Neuromorphic Computing Theory and System

