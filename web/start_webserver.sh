#!/bin/bash
source /home/$USER/miniconda3/bin/activate pds
python=$(which python)

if [ $# -eq 0 ]
  then
    echo "Usage: start_webserver.sh model1 model1_chktps_folder"
    exit
fi

$python dotsandboxesserver.py 8080 &>/dev/null &
FOO_PID=$!

echo "Server is running on port 8080 and agent on port 8081"
./dotsandboxesagent_az.py "configuration.$1" $2 10.0.0.11 8081

kill $FOO_PID 