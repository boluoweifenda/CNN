# TensorFlow-Slimmer image classification model library

A simple and extensible project based on 
[TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)
for 
- fast demonstrations of various ideas
- reimplementation of CNN architecture engineering papers, e.g., MobileNetV2, ShuffleNetV2.
- <a href="#exercise">exercises for Tsinghua lecture: Neuromorphic Computing Theory and System</a><br>

## Prerequisites
- A computer with NVIDIA GPUs
- CUDA + CuDNN
- TensorFlow (GPU version) tested: 1.14, 1.12
- python3
- tqdm

## Preparing datasets
Download and generate mnist, fashion and cifar datasets by:
```bash
cd data/xxx/
python3 download_and_convert_xxx.py
```
For imagenet dataset, please follow the instructions in 
[an automated script](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data),
and change the destinations in `source/datasets/dataset_factory` to the converted tf.record file locations


## Configuring training hyperparameters

Change or new an option file in `source/options/`, there are some templates for beginners.


### Training devices
Change the configuration in option file:

- Single GPU trainig with GPU-2: `gpu_list = [2]`
- Multi-GPU training with GPU-1 and GPU-3: `gpu_list = [1,3]`
- CPU-only training (deprecated): `gpu_list = []` 


### Network models

Change or new a model file in `source/nets/`, and register it in `nets_factory.py`

You can also modify and add other components, such as preprocessing, dataset. 
Don't forget to register them in individual factory files.

## Start training:
In case the option file name in `source/options/` is `your_options.py` 
```bash
cd source/
python3 top.py -o your_options
```
The training log files will be printed and saved in `log/time_your_title.txt`

You can check the training details in log files, and derive statistics for drawing curves.

And the model will be saved in `model/time_your_title` if you have configured the saving parameters.







## Exercise for Tsinghua lecture: Neuromorphic Computing Theory and System
<a id='exercise'></a>

完成下列实验，提交实验报告，内容包括：
- 实验整个流程关键信息的记录，分析，总结
- 运行、结果等截图，收敛曲线等等
- 个人感想

### Experiment 1:
使用MLP训练mnist，`/options/mlp_mnist.py` 记录：
- 收敛曲线（train正确率，test正确率，train loss）
- 尝试添加几层全连接，对比效果
- 尝试更改loss function 和 optimizer， 对比效果

### Experiment 2:
使用LeNet训练fashion，`/options/lenet_fashion.py` 记录:
- 收敛曲线（train正确率，test正确率，train loss）
- 尝试添加几层卷积，以及修改preprocess，对比效果
- 尝试使用MLP训练fashion，对比和CNN的差别

### Experiment 3:
- 参考[ResNet论文](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
- 补全`/options/resnet_cifar.py`中的设置
- 复现其ResNet-56的实验结果
