#!/bin/bash

# Nombre de machine ou NODES typiquement=1 sauf
#SBATCH -N 1

# Nombre de processus en general=1 (a mémoire distribues type miprun)
#SBATCH --ntasks=1

#SBATCH --gres=gpu:8

# Nom de votre job afficher dans la lise par squeue
#SBATCH --job-name=hazpi_train_80g

# Nom du fichier de sortie et des erreurs avec l'id du job
#####SBATCH --output=res_%j.log
#####SBATCH --error=res_%j.err

#####SBATCH --partition=gpu
#SBATCH --partition=gpu80G

# Mail pour etre informe de l'etat de votre job
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr

# Temps maximal d'execution du job ci dessous
# d-hh:mm:ss
#SBATCH --time=3-0:00:00

# Taille de la memoire exprime en Mega octets max=190000 ici 50G
#SBATCH --mem=50G

####SBATCH --exclude=node5
####SBATCH --nodelist=node6,node7

#set -o nounset
set -o errexit
set -o pipefail

echo "$0"


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/softsf2/anaconda/2021.05/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/softsf2/anaconda/2021.05/etc/profile.d/conda.sh" ]; then
        . "/home/softsf2/anaconda/2021.05/etc/profile.d/conda.sh"
    else
        export PATH="/home/softsf2/anaconda/2021.05/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
 
 # activate environments
#module load anaconda/3.2021.05
module load cuda/11.2

conda activate hazpi-fia2
#export LD_LIBRARY_PATH=/home/users/gdechalendar/cuda/lib64:${LD_LIBRARY_PATH}
/usr/bin/env python3 --version
if [[ -v SLURM_JOB_ID ]] ; then
    nvidia-smi
    
    # Affiche la (ou les gpus) allouee par Slurm pour ce job
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
        
echo "Begin on machine: `hostname`"
        
# conda list
        
##############
              
#echo 'Syncing data'
#install -d /scratch/gael/hazpi
#rsync -avz --inplace --delete-delay bergamote2-ib:/scratch_global/gael/concode/data/d_100k_762 /scratch/gael/concode/
#ls /scratch/gael/concode
              
echo 'Script starting'
              
cd /home/users/gdechalendar/Projets/HazPi/HazPi
              
CHECKPOINT_PATH=/home/users/gdechalendar/Projets/HazPi/checkpoints-fia2
install -d ${CHECKPOINT_PATH}
VOCAB_PATH=/home/users/gdechalendar/Projets/HazPi/vocab-fia2/
install -d ${VOCAB_PATH}

# run script
echo 'Training'
python train_transformer.py -data_path /home/users/jlopez/dataset/medical_articles.xlsx -checkpoint_path ${CHECKPOINT_PATH}  -vocab_save_dir ${VOCAB_PATH} -batch_size 32 -epochs 300 -no_filters

