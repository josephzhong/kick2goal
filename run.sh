#! /bin/bash

mkdir -p miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
bash miniconda3/miniconda.sh -b -u -p miniconda3
rm miniconda3/miniconda.sh
miniconda3/bin/conda create -y -n py39 python=3.9
miniconda3/envs/py39/bin/python -m pip install -r requirements.txt
miniconda3/envs/py39/bin/python -m pip install cmake
miniconda3/envs/py39/bin/python -m pip install stable-baselines3[extra]
mkdir models

# various commands needed to run your job
miniconda3/envs/py39/bin/python test_ppo.py $1 $2
mv tb_log/PPO_1/* ./
mv models/* ./