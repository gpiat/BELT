#!/bin/bash
#SBATCH -e test.err
#SBATCH -o test.out
# Choisir nb noeuds
#SBATCH -N 1
# Choisir partition (commande `sinfo` pour voir les differentes partitions)
#SBATCH --partition=gpu
# Nb GPU que je veux
#SBATCH --gres=gpu:1
# Mail pour etre informe de l'etat de votre job
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=guilhem.piat@cea.fr
# Nom de votre job afficher dans la lise par squeue
#SBATCH --job-name=transformer_test
# Limite du nb de CPU

# Chargment de vos modules
#
echo "Begin loading module..."
module load anaconda cuda/10.1

# Installation de torchtext si besoin
# TORCHTEXT_EXISTS=`pip freeze|grep torchtext`
# [ -z "$TORCHTEXT_EXISTS" ] && pip install --user torchtext

# # Update de pytorch si besoin
# PYTORCH_VERSION=`conda list | grep ^pytorch`
# PYTORCH_VERSION=`echo $PYTORCH_VERSION|cut -c 9-14`
# [ "$PYTORCH_VERSION" != "1.3.0" ] && conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# #[ "$PYTORCH_VERSION" != "1.3.0" ] && echo AAA

# Affiche la machine(s)
echo "Begin on machine :"
hostname
# nvidia-smi
echo $CUDA_VISIBLE_DEVICES


python -u ~/Documents/projects/BELT/evaluate.py

# python -u 

echo "Done."
# ./ Fin
