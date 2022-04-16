#!/bin/bash

### ::deriv.main

### ::prelude.main

if [ "${SLURM_CPUS_PER_TASK}" != "" ]; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

if [ "@BLOCK2" = "1" ]; then
    SCPT=dmrg
    if [ "@RESTART" = "1" ]; then
        SCPT=dmrg-rev
    fi
else
    SCPT=hife
fi

TJ=$(echo ${SCPT}.out.* | tr ' ' '\n' | grep '\*$' -v | wc -l)
export TJ=$(expr ${TJ} + 1)
echo ${SCPT}.out.${TJ} >> OUTFILE
echo $SLURM_JOBID >> JOBIDS

which orterun

if [ "$?" = "1" ] || [ "${SLURM_TASKS_PER_NODE}" = "" ]; then
    if [ "@BLOCK2" = "1" ]; then
        [ -f ./FCIDUMP ] && rm ./FCIDUMP
        ln -s @TMPDIR/FCIDUMP ./FCIDUMP
        cp @TMPDIR/${SCPT}.conf ${SCPT}.conf.${TJ}
        python3 -u $(which block2main) ${SCPT}.conf.${TJ} > ${SCPT}.out.${TJ}
    else
        python3 ${SCPT}.py @RESTART > ${SCPT}.out.${TJ}
    fi
else
    if [ "@BLOCK2" = "1" ]; then
        [ -f ./FCIDUMP ] && rm ./FCIDUMP
        ln -s @TMPDIR/FCIDUMP ./FCIDUMP
        cp @TMPDIR/${SCPT}.conf ${SCPT}.conf.${TJ}
        orterun --map-by ppr:$SLURM_TASKS_PER_NODE:node:pe=$OMP_NUM_THREADS \
            python3 -u $(which block2main) ${SCPT}.conf.${TJ} > ${SCPT}.out.${TJ}
    else
        orterun --map-by ppr:$SLURM_TASKS_PER_NODE:node:pe=$OMP_NUM_THREADS \
            python3 ${SCPT}.py @RESTART > ${SCPT}.out.${TJ}
    fi
fi

if [ "$?" = "0" ]; then
    echo "SUCCESSFUL TERMINATION"
else
    echo "ERROR TERMINATION"
fi