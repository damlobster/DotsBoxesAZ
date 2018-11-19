#!/bin/bash
source /home/$USER/miniconda3/bin/activate pds
python=$(which python)

if [ $# -eq 0 ]
  then
    echo "You must specify the folder of the trained models"
    exit
fi

$python dotsandboxesserver.py 8080 &>/dev/null &
FOO_PID=$!

echo "Server is running on port 8080 and agent on port 8081"
./dotsandboxesagent_az.py "configuration.resnet20" $1 10.0.0.11 8081

kill $FOO_PID 