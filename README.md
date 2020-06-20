# Augmented-Sliced-Wasserstein-Distances

This repository is to reproduce the experimental results in the paper **[Augmented Sliced Wasserstein Distances](https://arxiv.org/abs/2006.08812)**.
## Prerequisites

### Python packages
python==3.7.7

pytorch==1.4.0 

torchvision==0.5.0

cudatoolkit==10.1.243

cupy==7.40

numpy==1.18.1

pot==0.7.0

imageio=2.8.0

### Datasets

The CIFAR10 dataset will be automatically downloaded from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz when running the experiment on CIFAR10 dataset. 
The CELEBA dataset needs be be manually downloaded and can be found on the website http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, we use the cropped CELEBA dataset with 64x64 pixels.

### Precalculated Statistics for FID calculation

Precalculated statistics for datasets

- [cropped CelebA](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_celeba.npz) (64x64, calculated on all samples)
- [CIFAR 10](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz) (calculated on all training samples)

are provided at: http://bioinf.jku.at/research/ttur/.
## Project & Script Descriptions
The sliced Wasserstein flow example can be found in the [jupyter notebook](https://github.com/xiongjiechen/ASWD/blob/master/Sliced%20Waaserstein%20Flow.ipynb).

The following scripts belong to the generative modelling example:
- [main.py](https://github.com/xiongjiechen/ASWD/blob/master/main.py) : executable file.
- [utils.py](https://github.com/xiongjiechen/ASWD/blob/master/utils.py) : contains implementations of different sliced-based Wasserstein distances.
- [TransformNet.py](https://github.com/xiongjiechen/ASWD/blob/master/TransformNet.py.py) : architectures of neural networks used to map samples. 

## Experiment options
This code allows you to evaluate the performances of GANs trained with different variants of sliced-based Wasserstein distance on the CIFAR10 dataset. To train and evaluate the model, run the following command.

```
python main.py  --model-type ASWD --dataset CIFAR --epochs 200 --num-projection 1000 --batch-size 512 --lr 0.0005
```

## References 
### Code
The code of generative modelling example is based on the implementation of **[Distributional Sliced Wasserstein Distance](https://github.com/VinAIResearch/DSW)** by **[VinAI Research](https://github.com/VinAIResearch)**.

The pytorch code for calculating the **[Fréchet Inception Distance (FID score)](https://arxiv.org/abs/1706.08500)** is from https://github.com/mseitzer/pytorch-fid.

### Papers
This repo includes implemetations of the following sliced-based Wasserstein metric:
- [Distributional Sliced-Wasserstein and Applications to Generative Modeling](https://arxiv.org/pdf/2002.07367.pdf)
- [Generalized Sliced Wasserstein Distances](http://papers.nips.cc/paper/8319-generalized-sliced-wasserstein-distances)
- [Sliced Wasserstein Auto-Encoders](https://openreview.net/forum?id=H1xaJn05FQ)
- [Max-Sliced Wasserstein Distance and its Use for GANs](http://openaccess.thecvf.com/content_CVPR_2019/html/Deshpande_Max-Sliced_Wasserstein_Distance_and_Its_Use_for_GANs_CVPR_2019_paper.html)
