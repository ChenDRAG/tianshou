CUDA_VISIBLE_DEVICES=0 python step_pg.py \
--task "Ant-v3" \
--seed 0 \
--logdir "vpg_benchmark_log_norm_step" > vpg0Ant.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python step_pg.py \
--task "Ant-v3" \
--seed 1 \
--logdir "vpg_benchmark_log_norm_step" > vpg1Ant.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python step_pg.py \
--task "Ant-v3" \
--seed 2 \
--logdir "vpg_benchmark_log_norm_step" > vpg2Ant.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python step_pg.py \
--task "Ant-v3" \
--seed 3 \
--logdir "vpg_benchmark_log_norm_step" > vpg3Ant.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python step_pg.py \
--task "Ant-v3" \
--seed 4 \
--logdir "vpg_benchmark_log_norm_step" > vpg4Ant.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python step_pg.py \
--task "Ant-v3" \
--seed 5 \
--logdir "vpg_benchmark_log_norm_step" > vpg5Ant.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python step_pg.py \
--task "Ant-v3" \
--seed 6 \
--logdir "vpg_benchmark_log_norm_step" > vpg6Ant.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python step_pg.py \
--task "Ant-v3" \
--seed 7 \
--logdir "vpg_benchmark_log_norm_step" > vpg7Ant.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python step_pg.py \
--task "Ant-v3" \
--seed 8 \
--logdir "vpg_benchmark_log_norm_step" > vpg8Ant.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python step_pg.py \
--task "Ant-v3" \
--seed 9 \
--logdir "vpg_benchmark_log_norm_step" > vpg9Ant.txt 2>&1 &
