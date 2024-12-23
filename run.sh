#! /bin/bash

unzip py39.zip
cd kick2goal
../py39/bin/python -m pip install -r requirements.txt --no-cache-dir
#../py39/bin/python -m pip install cmake --no-cache-dir
../py39/bin/python -m pip install stable-baselines3[extra] --no-cache-dir
mkdir models

# various commands needed to run your job
../py39/bin/python $1 $2 $3 $4 $5
mv tb_log/*_1/* ../$2_$4.events
mv models/* ../
mv *.rewards ../