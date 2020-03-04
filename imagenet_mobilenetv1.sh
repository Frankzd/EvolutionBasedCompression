#!/bin/sh
mode=$1
path=$2
if [ "$mode" = "search" ];then
echo "Searching"
python ea.py /home/dianzi/SharedProject/DataSet/ImageNet --arch=mobilenet \
--lr=0.03 --validation-num=4  --batch-size=256 \
--resume=/home/dianzi/SharedProject/distiller/examples/classifier_compression/logs/baseline_mobilenet_cifar10/best.pth.tar \
--ea-cfg=./auto_compression_channels.yaml --ea-action-range 0 0.1 \
--ea-population-size=10 --ea-max-gen=40
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
if [ "$mode" = "train" ];then
echo "Training"
python ./classifier_compression/compress_classifier.py /home/dianzi/SharedProject/DataSet/ImageNet --arch=mobilenet \
--compress=
fi
