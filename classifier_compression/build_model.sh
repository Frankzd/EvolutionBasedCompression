#!/bin/bash
# train a mobilenet_v1 for cifar10
#python compress_classifier.py ${CIFAR10_PATH} --arch=mobilenet_cifar \
#    --epochs=120 --compress=./baseline_mobilenet_cifar.yaml


#train baseline models for cifar10
python compress_classifier.py ${CIFAR10_PATH} --arch=mobilenet_v2_cifar \
    --epochs=600 \
    --compress=./baseline_cifar.yaml \
    --workers=32 \
    --batch-size=256 \
    --lr=0.05 \
    --weight-decay=0.001 \
    --summary=model

#train baseline models for Imagenet(use pretrain models)
# python compress_classifier.py ${IMAGENET_PATH} --arch=mobilenet_v2 \
#     --pretrain \
#     --compress=./baseline_mobilenet.yaml \
#     --workers=16 \
#     --batch-size=128 \
#     --lr=0.001 \
#     --weight-decay=0.001 \
    # --evaluate
