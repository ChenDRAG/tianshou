#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python step_td3.py \
--task "HalfCheetah-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--logdir "log" > 1.txt 2>&1 &
sleep 10s

CUDA_VISIBLE_DEVICES=3 python step_td3.py \
--task "HalfCheetah-v3" \
--collect-per-step 8 \
--update-per-step 8 \
--training-num 4 \
--logdir "log" > 2.txt 2>&1 &
sleep 10s

CUDA_VISIBLE_DEVICES=1 python step_td3.py \
--task "HalfCheetah-v3" \
--collect-per-step 8 \
--update-per-step 8 \
--training-num 8 \
--logdir "log" > 3.txt 2>&1 &
sleep 10s

CUDA_VISIBLE_DEVICES=1 python step_td3.py \
--task "Ant-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--logdir "log" > 4.txt 2>&1 &
sleep 10s

CUDA_VISIBLE_DEVICES=2 python step_td3.py \
--task "Ant-v3" \
--collect-per-step 8 \
--update-per-step 8 \
--training-num 4 \
--logdir "log" > 5.txt 2>&1 &
sleep 10s

CUDA_VISIBLE_DEVICES=2 python step_td3.py \
--task "Ant-v3" \
--collect-per-step 8 \
--update-per-step 8 \
--training-num 8 \
--logdir "log" > 6.txt 2>&1 &


