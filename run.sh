#!/bin/bash
# SBATCH -N 1
# SBATCH -p main
# SBATCH -c 32
# SBATCH --gres=gpu:4
# SBATCH --nodelist=worker-4
# SBATCH --mail-user=zhang.dbs.lmu@proton.me
# SBATCH --mail-type=ALL

clear

hostname
nvidia-smi
which python

if [ "$(hostname)" = "plunder" ]; then
    ROOT_PATH=/zhome/wiss/zhang/
else
    ROOT_PATH=/home/wiss/zhang/
fi

source ${ROOT_PATH}/anaconda3/bin/activate ${ROOT_PATH}/anaconda3/envs/msclip
which python



export START_TIME=$(python -c "import datetime; print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))")

python -m torch.distributed.launch --nproc_per_node=4 --master_port=7013 \
main.py --train \
--config ${ROOT_PATH}/code/MS-CLIP/config_kat.yaml \
--backend 'gloo' 
