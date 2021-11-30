# GuideToGANs
A learner's guide to Generative Adversarial Networks (GAN) implemented by Pytorch in Python.
A lot of inspiration has been taken from @eriklindernoren 's repository right [here](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/README.md?plain=1).

## Table of Contents
  * [Installation](#installation)
  * [Implementation](#implementation)
    + [GAN](#gan)
    + [Deep Convolutional GAN](#deep-convolutional-gan)


## Installation
    $ git clone https://github.com/ayushvrma/GuideToGANs
    $ cd GuideToGANs/
    $ sudo pip install -r requirements.txt

## Implementations
### GAN
_Generative Adversarial Networks_

#### Abstract
We propose a new architecture for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G (basically decides whether the sample is real or fake). The training procedure for G is to maximize the probability of D making a mistake (to defeat the Discriminator). This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. We keep the Generator constant while training the Discriminator and vice versa. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.

[[Paper]](https://arxiv.org/abs/1406.2661) [[Code]](implementations/gan/gan.py)

#### Run Example
```
$ cd implementations/gan/
$ python3 gan.py
```

### Deep Convolutional GAN
_Deep Convolutional Generative Adversarial Networks_

#### Abstract
In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

[[Paper]](https://arxiv.org/abs/1511.06434) [[Code]](implementations/dcgan/dcgan.py)

#### Run Example
On MNIST dataset
```
$ cd implementations/dcgan/
$ python3 dcgan_mnist.py
```
On CIFAR10 dataset
```
$ cd implementations/dcgan/
$ python3 dcgan_cifar10.py
```