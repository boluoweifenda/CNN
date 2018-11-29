# Exercise for Tsinghua lecture: Neuromorphic Computing Theory and System

Code example for convolutional neural network training

## Prerequisites
- A computer with NVIDIA GPUs (Please, it is important)
- CUDA + CuDNN
- Tensorflow (GPU version)
- python3
- tqdm


## Data
Download and generate mnist and fashion datasets:
```bash
cd data/mnist/
python3 download_and_convert_mnist.py
```
and
```bash
cd data/fashion
python3 download_and_convert_fashion.py
```

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

And the model will be saved in model/xxxx if you have configured in opt.py

