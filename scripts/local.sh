#!/bin/bash
# set -x
if [ $# -lt 3 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi

ROLE=$1
shift

export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift
bin=$1
shift
arg="$@"
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8000

if [[ $ROLE == 'scheduler' ]]
then
    echo $ROLE
    # start the scheduler
    export DMLC_ROLE='scheduler'
    ${bin} ${arg} &
fi


if [[ $ROLE == 'server' ]]
then
    # start servers
    export DMLC_ROLE='server'
    # for ((count=0; count<${DMLC_NUM_SERVER}; count++)); do
    export HEAPPROFILE=./S${i}
    echo $ROLE
    ${bin} ${arg} &
    # done
fi

if [[ $ROLE == 'worker' ]]
then
    echo $ROLE
    # start workers
    export DMLC_ROLE='worker'
    # for ((i=0; i<${DMLC_NUM_WORKER}; ++i)); do
    export HEAPPROFILE=./W${i}
    ${bin} ${arg} &
    # done
fi

wait
