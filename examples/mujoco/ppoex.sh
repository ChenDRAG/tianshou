#!/bin/bash
echo "STARTED"
counted=0
selected_gpu=0

TXTLOGDIR="./temp0311/"
if [ ! -d $TXTLOGDIR ]; then
  mkdir $TXTLOGDIR
fi

main() {
    TASK_LIST=("HalfCheetah-v3")
    MAXSEED=10
    for TASK in ${TASK_LIST[*]}
    do
        for ((seed=0;seed<$MAXSEED;seed+=1))
        do
            select_gpu
            txtname=${TXTLOGDIR}${TASK}_`date '+%m-%d-%H-%M-%S'`_seed_${seed}.txt

            CUDA_VISIBLE_DEVICES=$selected_gpu python -u mujoco_ppo.py \
            --task $TASK \
            --hidden-sizes 64 64 \
            --target-kl 0 \
            --epoch 34 \
            --value-clip 0 \
            --batch-size 64 \
            --seed $seed \
            --logdir "no_rew_norm_no_vclip" > $txtname 2>&1 &
            
            sleep 2s
        done
    done
    for TASK in ${TASK_LIST[*]}
    do
        for ((seed=0;seed<$MAXSEED;seed+=1))
        do
            select_gpu
            txtname=${TXTLOGDIR}${TASK}_`date '+%m-%d-%H-%M-%S'`_seed_${seed}.txt

            CUDA_VISIBLE_DEVICES=$selected_gpu python -u mujoco_ppo.py \
            --task $TASK \
            --hidden-sizes 64 64 \
            --target-kl 0 \
            --epoch 34 \
            --batch-size 64 \
            --seed $seed \
            --logdir "no_rew_norm_with_vclip" > $txtname 2>&1 &
            
            sleep 2s
        done
    done
    echo "ENDED!"
}

function select_gpu(){
    MAX_NUM_PER_GPU=5
    MEMORY_THRESHOLD=70
    USAGE_THRESHOLD=80
    GPU_NUM=`nvidia-smi --query-gpu=count --format=csv,noheader,nounits -i 0`
    while true
    do
        for ((i=0;i<$GPU_NUM;i+=1))
        do
            ps=`nvidia-smi --query-compute-apps=pid -i $i --format=csv,noheader,nounits`
            ps=($ps)
            p_num=${#ps[*]}
            util=`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $i`
            mem_used=`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i`
            mem_total=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $i`
            let "memoryutil=mem_used*100/mem_total"
            if [ $p_num -ge $MAX_NUM_PER_GPU ]
            then
                echo "`date`       GPU $i is loaded with max number of processes."
                continue
            elif [ $memoryutil -gt $MEMORY_THRESHOLD ]
            then
                echo "`date`       GPU $i enough memory usage."
                continue
            elif [ $util -gt $USAGE_THRESHOLD ]
            then
                echo "`date`       GPU $i enough computation usage."
                continue
            else
                ((counted=counted+1))
                echo "`date`       GPU $i is selected to run the ${counted}th experiment!"
                echo "------------------------------------------------------------------"
                selected_gpu=`nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits -i $i`
                return
            fi
        done
        echo "`date`       sleeping, will reselect after 1 min. ${counted} experiments have started."
        sleep 1m
    done
}

main "$@"; exit