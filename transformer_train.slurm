#!/bin/bash
#SBATCH -e log_train.err
#SBATCH -o log_train.out
# Choisir nb noeuds
#SBATCH -N 1
# Choisir partition (commande `sinfo` pour voir les differentes partitions)
#SBATCH --partition=gpu
# Nb GPU que je veux
#SBATCH --gres=gpu:4
# Mail pour etre informe de l'etat de votre job
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=guilhem.piat@cea.fr
# Nom de votre job afficher dans la lise par squeue
#SBATCH --job-name=BELT_train
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

srun --gres=gpu:1 python -u ~/Documents/projects/BELT/train.py --epochs 10 --writepath /home/users/gpiat/Documents/projects/BELT/runs/train/run10 &
srun --gres=gpu:1 python -u ~/Documents/projects/BELT/train.py --epochs 50 --writepath /home/users/gpiat/Documents/projects/BELT/runs/train/run11 &
srun --gres=gpu:1 python -u ~/Documents/projects/BELT/train.py --epochs 100 --writepath /home/users/gpiat/Documents/projects/BELT/runs/train/run12 &
srun --gres=gpu:1 python -u ~/Documents/projects/BELT/train.py --epochs 500 --writepath /home/users/gpiat/Documents/projects/BELT/runs/train/run13 &
wait

# python -u 

echo "Done."
# ./ Fin
