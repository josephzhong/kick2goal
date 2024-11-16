#! /bin/bash


# various commands needed to run your job
./py39/bin/python kick2goal/test_ppo.py $1 $2
mv kick2goal/tb_log/PPO_1/* ./
mv kick2goal/models/* ./