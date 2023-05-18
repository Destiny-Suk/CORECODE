#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/span_extraction/train_QA.py \
    --model rob \
    --cuda 6 \
    --train_file dataset/span_extraction/train_task3.json\
    --validation_file dataset/span_extraction/dev_task3.json\
    --test_file dataset/span_extraction/test_task3.json
