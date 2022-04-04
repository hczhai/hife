
### start:deriv.main
#SBATCH --job-name @NAME
#SBATCH -o LOG.%j
#SBATCH --nodes=@NNODES
#SBATCH --time=@TIME
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=@NCORES
#SBATCH --no-requeue
#SBATCH --mem=@MEM
### end:deriv.main

### start:prelude.main
source ~/.bashrc

module purge
module load gcc/9.2.0

export PYSCF_TMPDIR=@TMPDIR

which python3
python3 --version
python3 -c "import pyscf; print(pyscf.__version__)"
python3 -c "import pyscf; print(pyscf.__file__)"
python3 -c "import pyscf; print(pyscf.lib.param.TMPDIR)"

echo SLURM_TASKS_PER_NODE=$SLURM_TASKS_PER_NODE
echo OMP_NUM_THREADS=$OMP_NUM_THREADS
echo SLURM_JOBID=$SLURM_JOBID
echo SLURM_JOB_NAME=$SLURM_JOB_NAME
echo HOST_NAME = $(hostname)
echo PWD = $(pwd)
echo SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

export PYTHONPATH=/home/hczhai/work/block2-old:$PYTHONPATH

### end:prelude.main
