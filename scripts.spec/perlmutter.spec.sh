
### start:deriv.main
#SBATCH --job-name @NAME
#SBATCH -o LOG.%j
#SBATCH -q @QUEUE
#SBATCH --nodes=@NNODES
#SBATCH --time=@TIME
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=@NCORES
#SBATCH --no-requeue
#SBATCH --constraint cpu
#SBATCH --mem=@MEM
### end:deriv.main

### start:prelude.main
source ~/.bashrc

X=/global/homes/h/hzhai/changroup/program/openmpi-4.1.2/install/bin
PATH=${PATH//":$X:"/":"}
PATH=${PATH/#"$X:"/}
PATH=${PATH/%":$X"/}
X=/global/homes/h/hzhai/changroup/program/openmpi-4.1.2/install/lib
LD_LIBRARY_PATH=${LD_LIBRARY_PATH//":$X:"/":"}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH/#"$X:"/}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH/%":$X"/}

module load cray-python/3.9.13.1
module load fast-mkl-amd/fast-mkl-amd
module load craype
module load PrgEnv-gnu/8.3.3
module load cray-pmi
module unload darshan

source /global/homes/h/hzhai/changroup/program/base-plmt/bin/activate

export PYTHONPATH=/global/homes/h/hzhai/changroup/program/block2/build-plmt:$PYTHONPATH
export PYSCF_TMPDIR=@TMPDIR

which python3
python3 --version
python3 -c "import pyscf; print(pyscf.__version__)"
python3 -c "import pyscf; print(pyscf.__file__)"
python3 -c "import pyscf; print(pyscf.lib.param.TMPDIR)"
python3 -c "import block2; print(block2.__file__)"

echo SLURM_TASKS_PER_NODE=$SLURM_TASKS_PER_NODE
echo OMP_NUM_THREADS=$OMP_NUM_THREADS
echo SLURM_JOBID=$SLURM_JOBID
echo SLURM_JOB_NAME=$SLURM_JOB_NAME
echo HOST_NAME = $(hostname)
echo PWD = $(pwd)
echo SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

export XRUN=srun
export PYSCF_MPIPREFIX=srun

### end:prelude.main
