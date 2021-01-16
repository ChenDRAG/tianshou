CUDA_VISIBLE_DEVICES=0 python step_sac.py \
--task "Humanoid-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 0 \
--logdir "sac_benchmark_log" > 115Humanoidfake.txt 2>&1 &        


CUDA_VISIBLE_DEVICES=1 python step_sac.py \
--task "Humanoid-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 1 \
--logdir "sac_benchmark_log" > 215Humanoidfake.txt 2>&1 &        


CUDA_VISIBLE_DEVICES=2 python step_sac.py \
--task "Humanoid-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 2 \
--logdir "sac_benchmark_log" > 315Humanoidfake.txt 2>&1 &        

CUDA_VISIBLE_DEVICES=3 python step_sac.py \
--task "Humanoid-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 3 \
--logdir "sac_benchmark_log" > 415Humanoidfake.txt 2>&1 &        



CUDA_VISIBLE_DEVICES=0 python step_sac.py \
--task "Humanoid-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 4 \
--logdir "sac_benchmark_log" > 515Humanoidfake.txt 2>&1 &        



CUDA_VISIBLE_DEVICES=1 python step_sac.py \
--task "Humanoid-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 5 \
--logdir "sac_benchmark_log" > 615Humanoidfake.txt 2>&1 &        



CUDA_VISIBLE_DEVICES=2 python step_sac.py \
--task "Humanoid-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 6 \
--logdir "sac_benchmark_log" > 715Humanoidfake.txt 2>&1 &        



CUDA_VISIBLE_DEVICES=3 python step_sac.py \
--task "Humanoid-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 7 \
--logdir "sac_benchmark_log" > 815Humanoidfake.txt 2>&1 &        



CUDA_VISIBLE_DEVICES=2 python step_sac.py \
--task "Humanoid-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 8 \
--logdir "sac_benchmark_log" > 915Humanoidfake.txt 2>&1 &        



CUDA_VISIBLE_DEVICES=3 python step_sac.py \
--task "Humanoid-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 9 \
--logdir "sac_benchmark_log" > 1015Humanoidfake.txt 2>&1 &        
