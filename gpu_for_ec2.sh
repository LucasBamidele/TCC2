sudo apt-get update -y
sudo apt-get upgrade -y linux-aws
sudo reboot
sudo apt-get install -y gcc make linux-headers-$(uname -r)

cat << EOF | sudo tee --append /etc/modprobe.d/blacklist.conf
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist nvidiafb
blacklist rivatv
EOF

#Edit the /etc/default/grub file and add the following line:
sudo nano /etc/default/grub 
GRUB_CMDLINE_LINUX="rdblacklist=nouveau"

sudo update-grub

# Download the driver package that you identified earlier as follows.

# For P2 and P3 instances, the following command downloads the NVIDIA driver,
# where xxx.xxx represents the version of the NVIDIA driver.

wget http://us.download.nvidia.com/tesla/418.87/nvidia-driver-local-repo-ubuntu1804-418.87.00_1.0-1_amd64.deb
sudo dpkg -i nvidia-driver-local-repo-ubuntu1804-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/nvidia-driver-local-repo-418.87.00/7fa2af80.pub
sudo dpkg -i nvidia-driver-local-repo-ubuntu1804-418.87.00_1.0-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-drivers
sudo reboot

nvidia-smi -q | head

git clone https://github.com/LucasBamidele/TCC2.git

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

tmux new -s mywindow

tmux a -t mywindow