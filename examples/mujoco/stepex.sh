sleep 7h
CUDA_VISIBLE_DEVICES=0 python step_td3.py \
--task "Ant-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 0 \
--logdir "stepcollector10seedexpclipdonemodify_1collect" > 1.txt 2>&1 &        
sleep 60s

CUDA_VISIBLE_DEVICES=1 python step_td3.py \
--task "Ant-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 1 \
--logdir "stepcollector10seedexpclipdonemodify_1collect" > 2.txt 2>&1 &        
sleep 60s


CUDA_VISIBLE_DEVICES=2 python step_td3.py \
--task "Ant-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 2 \
--logdir "stepcollector10seedexpclipdonemodify_1collect" > 3.txt 2>&1 &        
sleep 60s


CUDA_VISIBLE_DEVICES=3 python step_td3.py \
--task "Ant-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 3 \
--logdir "stepcollector10seedexpclipdonemodify_1collect" > 4.txt 2>&1 &        
sleep 60s


CUDA_VISIBLE_DEVICES=0 python step_td3.py \
--task "Ant-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 4 \
--logdir "stepcollector10seedexpclipdonemodify_1collect" > 5.txt 2>&1 &        
sleep 60s


CUDA_VISIBLE_DEVICES=1 python step_td3.py \
--task "Ant-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 5 \
--logdir "stepcollector10seedexpclipdonemodify_1collect" > 6.txt 2>&1 &        
sleep 60s


CUDA_VISIBLE_DEVICES=2 python step_td3.py \
--task "Ant-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 6 \
--logdir "stepcollector10seedexpclipdonemodify_1collect" > 7.txt 2>&1 &        
sleep 60s


CUDA_VISIBLE_DEVICES=3 python step_td3.py \
--task "Ant-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 7 \
--logdir "stepcollector10seedexpclipdonemodify_1collect" > 8.txt 2>&1 &        
sleep 60s


CUDA_VISIBLE_DEVICES=3 python step_td3.py \
--task "Ant-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 8 \
--logdir "stepcollector10seedexpclipdonemodify_1collect" > 9.txt 2>&1 &        
sleep 60s


CUDA_VISIBLE_DEVICES=2 python step_td3.py \
--task "Ant-v3" \
--collect-per-step 1 \
--update-per-step 1 \
--training-num 1 \
--seed 9 \
--logdir "stepcollector10seedexpclipdonemodify_1collect" > 10.txt 2>&1 &        
sleep 60s

