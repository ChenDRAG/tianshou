CUDA_VISIBLE_DEVICES=0 python step_ddpg.py \
--task "InvertedPendulum-v2" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 0 \
--hidden-layer-size 256 256 \
--batch-size 256 \
--logdir "ddpgbenchmark" > 1InvertedPendulum.txt 2>&1 &        

CUDA_VISIBLE_DEVICES=1 python step_ddpg.py \
--task "InvertedPendulum-v2" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 1 \
--hidden-layer-size 256 256 \
--batch-size 256 \
--logdir "ddpgbenchmark" > 2InvertedPendulum.txt 2>&1 &        


CUDA_VISIBLE_DEVICES=2 python step_ddpg.py \
--task "InvertedPendulum-v2" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 2 \
--hidden-layer-size 256 256 \
--batch-size 256 \
--logdir "ddpgbenchmark" > 3InvertedPendulum.txt 2>&1 &        


CUDA_VISIBLE_DEVICES=3 python step_ddpg.py \
--task "InvertedPendulum-v2" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 3 \
--hidden-layer-size 256 256 \
--batch-size 256 \
--logdir "ddpgbenchmark" > 4InvertedPendulum.txt 2>&1 &        


CUDA_VISIBLE_DEVICES=0 python step_ddpg.py \
--task "InvertedPendulum-v2" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 4 \
--hidden-layer-size 256 256 \
--batch-size 256 \
--logdir "ddpgbenchmark" > 5InvertedPendulum.txt 2>&1 &        


CUDA_VISIBLE_DEVICES=1 python step_ddpg.py \
--task "InvertedPendulum-v2" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 5 \
--hidden-layer-size 256 256 \
--batch-size 256 \
--logdir "ddpgbenchmark" > 6InvertedPendulum.txt 2>&1 &        


CUDA_VISIBLE_DEVICES=2 python step_ddpg.py \
--task "InvertedPendulum-v2" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 6 \
--hidden-layer-size 256 256 \
--batch-size 256 \
--logdir "ddpgbenchmark" > 7InvertedPendulum.txt 2>&1 &        


CUDA_VISIBLE_DEVICES=3 python step_ddpg.py \
--task "InvertedPendulum-v2" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 7 \
--hidden-layer-size 256 256 \
--batch-size 256 \
--logdir "ddpgbenchmark" > 8InvertedPendulum.txt 2>&1 &        


CUDA_VISIBLE_DEVICES=0 python step_ddpg.py \
--task "InvertedPendulum-v2" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 8 \
--hidden-layer-size 256 256 \
--batch-size 256 \
--logdir "ddpgbenchmark" > 9InvertedPendulum.txt 2>&1 &        


CUDA_VISIBLE_DEVICES=1 python step_ddpg.py \
--task "InvertedPendulum-v2" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 9 \
--hidden-layer-size 256 256 \
--batch-size 256 \
--logdir "ddpgbenchmark" > 10InvertedPendulum.txt 2>&1 &        

