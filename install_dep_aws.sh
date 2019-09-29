#!/bin/bash

RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NO_COLOR='\033[0m'

dependency=(
    "sudo apt install python3-pip"
    "sudo pip3 install numpy"
    "sudo pip3 install pygame"
    "sudo pip3 install box2d-py"
    "sudo pip3 install pygame-menu"
    "sudo pip3 install tensorflow-gpu"
    "sudo pip3 install keras"
    "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb"
    "sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb"
    "sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub"
    "sudo apt-get update"
    "wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb"
    "sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb"
    "sudo apt-get update"
# Install NVIDIA driver
    "sudo apt-get install --no-install-recommends nvidia-driver-418"
# Reboot. Check that GPUs are visible using the command: nvidia-smi
)

result(){
    if [ $? == 0 ]; then
        printf "${GREEN}######### ---- Success ---- #########${NO_COLOR}\n\n"
    else
        printf "${RED}######### ---- Failure ---- #########${NO_COLOR}\n\n" 
    fi
} 

install-necessary(){
    declare -a argAry=("${!1}")
    for i in "${argAry[@]}"
    do
        $i      # Run the command
        result  # Show the result of the operation
    done
}


printf "${BLUE}######### ---- Updating packages... ---- #########${NO_COLOR}\n"
sudo apt-get update
result

install-necessary dependency[@]
