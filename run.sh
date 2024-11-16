#! /bin/bash

unzip miniconda3.zip
cd kick2goal
../miniconda3/bin/conda create -y -n py39 python=3.9
../miniconda3/envs/py39/python -m pip install -r requirements.txt
../miniconda3/envs/py39/python -m pip install stable-baselines3[extra]

# various commands needed to run your job

mkdir models
../miniconda3/envs/py39/python test_ppo.py $1 $2
mv tb_log/PPO_1/* ../
mv models/* ../