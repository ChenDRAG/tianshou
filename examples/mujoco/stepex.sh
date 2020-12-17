CUDA_VISIBLE_DEVICES=0 python mujoco_td3.py \
--task "HalfCheetah-v3" \
--seed 0 \
--logdir "log" > 1.txt 2>&1 &
sleep 60s


CUDA_VISIBLE_DEVICES=1 python mujoco_td3.py \
--task "HalfCheetah-v3" \
--seed 1 \
--logdir "log" > 2.txt 2>&1 &
sleep 60s



CUDA_VISIBLE_DEVICES=2 python mujoco_td3.py \
--task "HalfCheetah-v3" \
--seed 2 \
--logdir "log" > 3.txt 2>&1 &
sleep 60s



CUDA_VISIBLE_DEVICES=3 python mujoco_td3.py \
--task "HalfCheetah-v3" \
--seed 3 \
--logdir "log" > 4.txt 2>&1 &
sleep 60s


CUDA_VISIBLE_DEVICES=0 python mujoco_td3.py \
--task "Ant-v3" \
--seed 0 \
--logdir "log" > 5.txt 2>&1 &
sleep 60s



CUDA_VISIBLE_DEVICES=1 python mujoco_td3.py \
--task "Ant-v3" \
--seed 1 \
--logdir "log" > 6.txt 2>&1 &
sleep 60s



CUDA_VISIBLE_DEVICES=2 python mujoco_td3.py \
--task "Ant-v3" \
--seed 2 \
--logdir "log" > 7.txt 2>&1 &
sleep 60s



CUDA_VISIBLE_DEVICES=3 python mujoco_td3.py \
--task "Ant-v3" \
--seed 3 \
--logdir "log" > 8.txt 2>&1 &
sleep 60s

