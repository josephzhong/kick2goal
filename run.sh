#! /bin/bash


# various commands needed to run your job
./py39/bin/python test_ppo.py $1 $2
mv tb_log/PPO_1/* ./
mv models/* ./