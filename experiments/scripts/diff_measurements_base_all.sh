#!/usr/bin/env bash

if [ -z "$1" ] || [ -z "$2" ];
then
    echo "Usage: $0 <path_to_cfg> <comma-separated number of measurements> <extra arguments (optional)>";
    exit 1;
fi

set -x
set -e
CFG=$1
CFG_STUB=(${CFG//\// })
LEN_CFG=${#CFG_STUB[@]}
CFG_STUB=${CFG_STUB[(( LEN_CFG - 1 ))]}
CFG_STUB=${CFG_STUB//\.yml/}
mkdir -p debug/${CFG_STUB}

MS=$2
MS=(${MS//,/ })

GPU_NUM=$(( `nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version --format=csv | wc -l` - 1 ))
LEN_M=${#MS[@]}
PER_DEVICE=$(( $LEN_M % $GPU_NUM ))

start=2
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:$start:$len}

I=0
RUN_PIDS=""
for m in "${MS[@]}"
do
    CUDA_VISIBLE_DEVICES=$(( $I % $GPU_NUM ))
    I=$(( I + 1 ))
    export CUDA_VISIBLE_DEVICES
    DEBUG_FILE_PATH="debug/${CFG_STUB}/iters_${m}_train.log"
    python main.py --cfg $CFG --is_train \
   --default_test_params --reconstruction_res --cs_num_measurements $m \
   --test_batch_size 20 $EXTRA_ARGS &> $DEBUG_FILE_PATH &
   CUR_PID=$!
   RUN_PIDS="$CUR_PID $RUN_PIDS"
done

for PID in $RUN_PIDS;
do
    wait $PID;
done