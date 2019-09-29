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
    # "sudo pip3 install tensorflow-gpu"
    # "sudo pip3 install keras"
    # "sudo apt-get install build-essential dkms"
    # "sudo apt-get install freeglut3 freeglut3-dev libxi-dev libxmu-dev"
    # "sudo apt-get update -y"
    # "sudo apt-get upgrade -y linux-aws"
#     "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb"
#     "sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb"
#     "sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub"
#     "sudo apt-get update"
#     "wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb"
#     "sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb"
#     "sudo apt-get update"
# # Install NVIDIA driver
#     "sudo apt-get install --no-install-recommends nvidia-driver-418"
# Reboot. Check that GPUs are visible using the command: nvidia-smi

#run the following:

# sudo apt install cuda9.0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 \
#     cuda-cusolver-9-0 cuda-cusparse-9-0 libcudnn7=7.2.1.38-1+cuda9.0 \
#     libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
# sudo apt-get update
# sudo apt-get -y install cuda

#wget http://us.download.nvidia.com/tesla/418.87/nvidia-driver-local-repo-ubuntu1804-418.87.00_1.0-1_amd64.deb

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
