#! /bin/bash

# various commands needed to run your job
python test_ppo.py $1 $2
mv tb_log/PPO_1/* ./
mv tb_