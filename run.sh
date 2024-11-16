#! /bin/bash

python -m venv py39
source ./py39/bin/activate;python -m pip install -r requirements.txt
source ./py39/bin/activate;python -m pip install stable-baselines3[extra]
mkdir models

# various commands needed to run your job
source ./py39/bin/activate;python test_ppo.py $1 $2
mv tb_log/PPO_1/* ./
mv models/* ./