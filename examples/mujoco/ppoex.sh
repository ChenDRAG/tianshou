CUDA_VISIBLE_DEVICES=0 python mujoco_ppo.py \
--task "HalfCheetah-v3" \
--hidden-sizes 64 64 \
--step-per-collect 2048 \
--batch-size 64 \
--seed 0 \
--logdir "ppo_official3" > 115HalfCheetahfake.txt 2>&1 &        


CUDA_VISIBLE_DEVICES=1 python mujoco_ppo.py \
--task "HalfCheetah-v3" \
--hidden-sizes 64 64 \
--step-per-collect 2048 \
--batch-size 64 \
--seed 1 \
--logdir "ppo_official3" > 215HalfCheetahfake.txt 2>&1 &        


CUDA_VISIBLE_DEVICES=2 python mujoco_ppo.py \
--task "HalfCheetah-v3" \
--hidden-sizes 64 64 \
--step-per-collect 2048 \
--batch-size 64 \
--seed 2 \
--logdir "ppo_official3" > 315HalfCheetahfake.txt 2>&1 &        

CUDA_VISIBLE_DEVICES=3 python mujoco_ppo.py \
--task "HalfCheetah-v3" \
--hidden-sizes 64 64 \
--step-per-collect 2048 \
--batch-size 64 \
--seed 3 \
--logdir "ppo_official3" > 415HalfCheetahfake.txt 2>&1 &        



CUDA_VISIBLE_DEVICES=0 python mujoco_ppo.py \
--task "HalfCheetah-v3" \
--hidden-sizes 64 64 \
--step-per-collect 2048 \
--batch-size 64 \
--seed 4 \
--logdir "ppo_official3" > 515HalfCheetahfake.txt 2>&1 &        



CUDA_VISIBLE_DEVICES=1 python mujoco_ppo.py \
--task "HalfCheetah-v3" \
--hidden-sizes 64 64 \
--step-per-collect 2048 \
--batch-size 64 \
--seed 5 \
--logdir "ppo_official3" > 615HalfCheetahfake.txt 2>&1 &        



CUDA_VISIBLE_DEVICES=2 python mujoco_ppo.py \
--task "HalfCheetah-v3" \
--hidden-sizes 64 64 \
--step-per-collect 2048 \
--batch-size 64 \
--seed 6 \
--logdir "ppo_official3" > 715HalfCheetahfake.txt 2>&1 &        



CUDA_VISIBLE_DEVICES=3 python mujoco_ppo.py \
--task "HalfCheetah-v3" \
--hidden-sizes 64 64 \
--step-per-collect 2048 \
--batch-size 64 \
--seed 7 \
--logdir "ppo_official3" > 815HalfCheetahfake.txt 2>&1 &        



CUDA_VISIBLE_DEVICES=2 python mujoco_ppo.py \
--task "HalfCheetah-v3" \
--hidden-sizes 64 64 \
--step-per-collect 2048 \
--batch-size 64 \
--seed 8 \
--logdir "ppo_official3" > 915HalfCheetahfake.txt 2>&1 &        



CUDA_VISIBLE_DEVICES=3 python mujoco_ppo.py \
--task "HalfCheetah-v3" \
--hidden-sizes 64 64 \
--step-per-collect 2048 \
--batch-size 64 \
--seed 9 \
--logdir "ppo_official3" > 1015HalfCheetahfake.txt 2>&1 &        
