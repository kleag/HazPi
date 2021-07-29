#!/bin/bash

# Nombre de machine ou NODES typiquement=1 sauf
#SBATCH -N 1

# Nombre de processus en general=1 (a mémoire distribues type miprun)
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:V100-SXM2-32GB:4

# -mem=<size[units]>
#     Specify the real memory required per node. Default units are megabytes. Different units can be specified using the suffix [K|M|G|T]

# Nom de votre job afficher dans la lise par squeue
#SBATCH --job-name=hazpi_train

# Nom du fichier de sortie et des erreurs avec l'id du job
#SBATCH --output=res_%j.log
#####SBATCH --error=res_%j.err

##SBATCH --partition=lasti,gpu,gpuv100,gpup100,gpup6000
#SBATCH --partition=lasti,gpup6000

# Mail pour etre informe de l'etat de votre job
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr

# Temps maximal d'execution du job ci dessous
#SBATCH --time=12:00:00

# Taille de la memoire exprime en Mega octets max=190000 ici 50G
#SBATCH --mem=50G

set -o nounset
set -o errexit
set -o pipefail

echo "$0"

/usr/bin/env python3 --version
if [[ -v SLURM_JOB_ID ]] ; then
  nvidia-smi

  # Affiche la (ou les gpus) allouee par Slurm pour ce job
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

echo "Begin on machine: `hostname`"

conda list
module load cuda/10.1

EXECUTOR=srun
if [ -z ${SLURM_JOB_ID+x} ]; then
  echo "Not in sbatch"
  EXECUTOR=bash
fi
echo "EXECUTOR is ${EXECUTOR}"

cd /home/users/gdechalendar/Projets/NewsGene/HazPi
${EXECUTOR} python train_transformer.py -data_path /home/users/jlopez/dataset/medical_articles.xlsx -checkpoint_path models/checkpoints/ -vocab_save_dir models/vocab/ -batch_size 24 -epochs 300 -no_filters
wait

echo "Done."

