# **ISMALLNET: LABEL DECOUPLING-BASED DENSELY NESTED NETWORK FOR INFRARED SMALL TARGET DETECTION**

by Zhiheng Hu, Yongzhen Wang, Peng Li, Jie Qin, Mingqiang Wei

# Introduction
![image-20221021204340653](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20221021204340653.png)

Small targets are often submerged in cluttered backgrounds of infrared images.
Conventional detectors tend to generate false alarms, while CNN-based detectors lose small targets in deep layers.
To handle above issues, we propose iSmallNet, a label decoupling-based densely nested network for infrared small object detection, which is designed as a three-stream neural network.
To fully capture the shape information of small targets, we exploit label decoupling to decompose the original labeled ground-truth (GT) map into an interior map and a boundary map.
The GT map, in collaboration with the two additional maps, breaks the unbalanced distribution of small object boundaries.
Besides the novel architecture of iSmallNet, we have two key contributions to improve the detection ability of infrared small objects.
First, to maintain small targets in deep layers, we develop a multi-scale nested interactive module to explore a wider range of context information.
Second, we develop an interior-boundary fusion module to integrate multi-granularity information to further enhance the detection performance.
Experiments on both the NUAA-SIRST and NUDT-SIRST datasets show clear improvements for iSmallNet over 11 state-of-the-art detectors.

# Prerequisites


* [Python 3.5](https://www.python.org/)

* [Pytorch 1.3](https://pytorch.org/)

* [OpenCV](https://opencv.org/)

* [Numpy](https://numpy.org/)

* [TensorboardX](https://github.com/lanpa/tensorboardX)

* [Tqdm](https://github.com/tqdm/tqdm)

* [Apex](https://github.com/NVIDIA/apex)

# Datasets
Download the following datasets and unzip them into `data` folder.

* [NUAA-SIRST](https://github.com/YimianDai/sirst)

* [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection)

# Usage
**1. Label Decouping**

```
python utils.py --dataset [dataset-name]
```

**2. Train**

```
python train.py --dataset [dataset-name]
```

**3. Test**

```
python test.py --dataset [dataset-name]
```

**4. Eval**

```
python eval.py --dataset [dataset-name]
```

# Results and Trained Models

![image-20221021204452533](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20221021204452533.png)
