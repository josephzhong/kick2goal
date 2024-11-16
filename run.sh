#! /bin/bash

python -m venv py39
./py39/bin/python -m pip install -r requirements.txt
./py39/bin/python -m pip install stable-baselines3[extra]
mkdir models

# various commands needed to run your job
./py39/bin/python test_ppo.py $1 $2
mv tb_log/PPO_1/* ./
mv models/* ./