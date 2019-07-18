clear;
cp -rf cifar10_baseline.h5 read_cifar10_baseline.h5
python h5read.py
python test_cifar10.py
