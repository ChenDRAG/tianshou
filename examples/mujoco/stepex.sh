#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python step_td3.py \
--task "HalfCheetah-v3" \
--logdir "log" > 1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python step_td3.py \
--task "HalfCheetah-v3" \
--logdir "log" > 2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python step_td3.py \
--task "HalfCheetah-v3" \
--logdir "log" > 3.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python step_td3.py \
--task "Ant-v3" \
--logdir "log" > 4.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python step_td3.py \
--task "Ant-v3" \
--logdir "log" > 5.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python step_td3.py \
--task "Ant-v3" \
--logdir "log" > 6.txt 2>&1 &


