!#/bin/bash

mkdir -p $@/Datasets/CIFAR/
cd $@/Datasets/CIFAR/
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
cd ..
