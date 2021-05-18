# A Coarse-to-Fine Two-Stage Attentive Network for Haze Removal of Remote Sensing Images

[Paper Link](https://ieeexplore.ieee.org/document/9136742)

Yufeng Li, and [Xiang Chen](https://cxtalk.github.io/)

## Abstract
In many remote sensing (RS) applications, haze seriously degrades the quality of optical RS images and even brings inconvenience to the following high-level visual tasks such as RS detection. In this letter, we address this challenge by designing a first-coarse-then-fine two-stage dehazing neural network, named FCTF-Net. The structure is simple but effective: the first stage of image dehazing extracts multiscale features through the encoder-decoder architecture and, therefore, allows the second stage of dehazing for better refining the results of the previous stage. In addition, we combine the channel attention mechanism with the basic convolution block, considering that different channel characteristics contain entirely different weighting information, to effectively deal with irregular distribution of haze in RS images. Owing to the scarcity of various and quality hazy RS data sets, we adopt two different synthesis methods to generate large-scale image pairs for uniform and nonuniform hazy images. This two-stage network, when trained in an end-to-end fashion, yields the state-of-the-art performances on both the synthetic data sets and real-world images with more visually pleasing dehazed results. Both the synthetic data set and the code are publicly available at https://github.com/cxtalk/FCTF-Net.

## Requirements
- CUDA 9.0
- Python 3.7
- Pytorch 1.4.0
- Torchvision 0.2.0

## Dataset
TO DO

## Citation
```
@ARTICLE{9136742,  
author={Y. {Li} and X. {Chen}},  
journal={IEEE Geoscience and Remote Sensing Letters},   
title={A Coarse-to-Fine Two-Stage Attentive Network for Haze Removal of Remote Sensing Images},   
year={2020},  
volume={},  
number={},  
pages={1-5},  
doi={10.1109/LGRS.2020.3006533}}
```

## Contact

If you are interested in our work or have any questions, please directly contact me. Email: cv.xchen@gmail.com

## Acknowledgments

Codes are heavily borrowed from [GridDehazeNet](https://github.com/proteus1991/GridDehazeNet).
