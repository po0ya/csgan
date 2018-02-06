#!/usr/bin/env bash


if [ -z "$1" ] || [ -z "$2" ];
then
    echo "Usage: $0 <classifier config> <path to feature file> <extra arguments (optional)>";
    exit 1;
fi

set -x
CFG=$1
FILE_PATH=$2
start=2
array=( $@ )r
len=${#array[@]}
EXTRA_ARGS=${array[@]:$start:$len}
EXTRA_ARGS=${array[@]:$start:$len}

EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
EXP_NAME_CFG=`echo $CFG | grep -oP '\/\K(\w|\.)+(?=.yml)'`
LOG="experiments/logs/${EXP_NAME_CFG}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python classify.py --cfg $CFG --feature_file $FILE_PATH $EXTRA_ARGS
