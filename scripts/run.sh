#!/bin/bash

### ::deriv.main

### ::prelude.main

TJ=$(echo hife.out.* | tr ' ' '\n' | grep '\*$' -v | wc -l)
export TJ=$(expr ${TJ} + 1)
echo hife.out.${TJ} >> OUTFILE
echo $SLURM_JOBID >> JOBIDS

if [ "${SLURM_CPUS_PER_TASK}" != "" ]; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

which orterun

if [ "$?" = "1" ] || [ "${SLURM_TASKS_PER_NODE}" = "" ]; then
    python3 hife.py @RESTART > hife.out.${TJ}
else
    orterun --map-by ppr:$SLURM_TASKS_PER_NODE:node:pe=$OMP_NUM_THREADS \
        python3 hife.py @RESTART > hife.out.${TJ}
fi

if [ "$?" = "0" ]; then
    echo "SUCCESSFUL TERMINATION"
else
    echo "ERROR TERMINATION"
fi