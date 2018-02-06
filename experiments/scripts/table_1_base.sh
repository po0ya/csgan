#!/usr/bin/env bash

if [ -z "$1" ] || [ -z "$2" ];
then
    echo "Usage: $0 <path to cfg> <number of measurements> <comma-separated number of original samples> <extra arguments (optional)>";
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
NUM_MEASUREMENTS=$2
ORIGS=$3
ORIGS=(${ORIGS//,/ })

GPU_NUM=$(( `nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version --format=csv | wc -l` - 1 ))
LEN_M=${#ORIGS[@]}
PER_DEVICE=$(( $LEN_M % $GPU_NUM ))

start=3
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:$start:$len}

I=0
RUN_PIDS=""
for O in "${ORIGS[@]}"
do
    CUDA_VISIBLE_DEVICES=$(( $I % $GPU_NUM ))
    I=$(( $I + 1 ))
    export CUDA_VISIBLE_DEVICES
    DEBUG_FILE_PATH="debug/${CFG_STUB}/origs_${O}_train.log"
    python main.py --cfg $CFG --is_train \
    --test_results_split test \
   --default_test_params --reconstruction_res --num_feats 1000 --cs_num_measurements $NUM_MEASUREMENTS --orig_size $O \
   --test_batch_size 20 $EXTRA_ARGS &> $DEBUG_FILE_PATH  &
   CUR_PID=$!
   RUN_PIDS="$CUR_PID $RUN_PIDS"
done

for PID in $RUN_PIDS;
do
    wait $PID;
done