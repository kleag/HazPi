#!/bin/bash

# Nombre de machine ou NODES typiquement=1 sauf
#SBATCH -N 1

# Nombre de processus en general=1 (a mémoire distribues type miprun)
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1

# Nom de votre job afficher dans la lise par squeue
#SBATCH --job-name=hazpi_summarize

# Nom du fichier de sortie et des erreurs avec l'id du job
#####SBATCH --output=res_%j.log
#####SBATCH --error=res_%j.err

#SBATCH --partition=gpu,gpuv100,gpup6000,lasti

# Mail pour etre informe de l'etat de votre job
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr

# Temps maximal d'execution du job ci dessous
# d-hh:mm:ss
#SBATCH --time=5:00:00

# Taille de la memoire exprime en Mega octets max=190000 ici 50G
#SBATCH --mem=50G

####SBATCH --exclude=node5
####SBATCH --nodelist=node6,node7

#set -o nounset
set -o errexit
set -o pipefail

echo "$0"



# activate environments
source /softs/anaconda/3-5.3.1/bin/activate
source activate hazpi
module load cuda/10.1
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

# run script
echo 'Summarizing'

python summarize.py -checkpoint_path /home/data/jlopez/experiments_all_db/ckp_not_all_vocab/ckp_pre_transformer_4Layers_moreEnc_vocab100_100/epoch_350/fine-tuning/epoch_550/epoch_550_FT_1/ -path_summaries_encoded /home/data/jlopez/headlines/results_complete_db/test/dec_bs_fine-tuning_epochs_350_550_4L_4E_noFilters_encoder2000_FT1/encoded_ngram/ -path_summaries_decoded /home/data/jlopez/headlines/results_complete_db/test/dec_bs_fine-tuning_epochs_350_550_4L_4E_noFilters_encoder2000_FT1/decoded_ngram/ -path_summaries_error /home/data/jlopez/headlines/results_complete_db/test/dec_bs_fine-tuning_epochs_350_550_4L_4E_noFilters_encoder2000_FT1/error_ngram/ -batch_size 32 -num_heads 8 -dff 2048 -num_layers 4 -d_model 128 -encoder_max_vocab 100000 -decoder_max_vocab 100000 -num_layers 4 -vocab_load_dir /home/data/jlopez/experiments_all_db/vocab/ckp_pre_transformer_4Layers_moreEnc_vocab100_100/ -ngram_size 2 -k 6 -len_summary 216 file.txt

