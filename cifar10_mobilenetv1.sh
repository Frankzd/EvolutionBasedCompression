#!/bin/sh
mode=$1
path=$2
if [ "$mode" = "search" ];then
echo "Searching"
python ea.py /home/dianzi/SharedProject/DataSet/Cifar --arch=mobilenet_v2_cifar \
    --lr=0.03 --validation-num=3  --batch-size=512 \
    --resume=/home/dianzi/SharedProject/EvolitionBasedCompression/classifier_compression/logs/baseline_mobilenet_v2_cifar10/best.pth.tar \
    --ea-cfg=./auto_compression_channels.yaml --ea-action-range 0.2 0.8 \
    --ea-population-size=20 --ea-max-gen=50 --ea-ft-epochs=0 \
    --ea-target-density=0.3
fi
if [ "$mode" = "eval" ];then
echo "Evaluating"
python ./classifier_compression/compress_classifier.py /home/dianzi/SharedProject/DataSet/Cifar --arch=mobilenet_cifar \
    --resume=$path \
    --evaluate 
fi
if [ "$mode" = "finetune" ];then
echo "Fine-tuning"
python ./classifier_compression/compress_classifier.py /home/dianzi/SharedProject/DataSet/Cifar --arch=mobilenet_cifar \
    --resume=$path --lr=0.001 --epoch=10
fi
