#!/bin/bash
 
#! Name of the job:
#SBATCH -J hsGRN_Gillespie

#! Which project should be charged:
#SBATCH -A LOCKE-SL3-CPU
#SBATCH -p icelake

#! How many (MPI) tasks will there be in total? (<= nodes*76)
#! The Ice Lake (icelake) nodes have 76 CPUs (cores) each and
#! 3380 MiB of memory per CPU.
#SBATCH --ntasks=1

#SBATCH -D /home/jz531/rds/hpc-work/GRN_heatshock_At/
#SBATCH -o /home/jz531/rds/hpc-work/hpc_output/Gilespie_output.log
#SBATCH -c 1     # increase if doing multiprocessing, max 32 CPUs
#SBATCH --array=1 # max is 9999
#SBATCH --mem-per-cpu=5980MB   # max 5980MB or 12030MB for skilake-himem

#! How much wallclock time will be required?
#SBATCH --time 2:00:00            # HH:MM:SS with maximum 12:00:00 for SL3 or 36:00:00 for SL2


#! prevent the job from being requeued (e.g. if interrupted by node failure or system downtime):
#SBATCH --no-requeue
 
. /etc/profile.d/modules.sh # Leave this line (enables the module command)
module purge  # Removes all modules still loaded
module load miniconda/3
source /home/jz531/.bashrc
conda activate /home/jz531/.conda/envs/model_GRN
conda list

CMD="python3 HSPR_AZ_v6.py -nit 30 -tsp 20000 -hss 10000 -hsd 5000"

eval $CMD