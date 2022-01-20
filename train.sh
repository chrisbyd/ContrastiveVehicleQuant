#!/bin/bash
# CUDA_VISIBLE_DEVICES=1 python main.py -c configs/veri_1024.yml

CUDA_VISIBLE_DEVICES=1 python main.py -c configs/veri_512.yml

CUDA_VISIBLE_DEVICES=1 python main.py -c configs/veri_256.yml