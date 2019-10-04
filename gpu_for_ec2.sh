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


# wget http://us.download.nvidia.com/tesla/410.104/NVIDIA-Linux-x86_64-410.104.run
# sudo sh ./NVIDIA-Linux-x86_64-410.104.run --no-drm --disable-nouveau --dkms --silent --install-libglvnd

wget http://us.download.nvidia.com/tesla/410.129/nvidia-diag-driver-local-repo-ubuntu1804-410.129_1.0-1_amd64.deb
sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1804-410.129_1.0-1_amd64.deb
sudo apt-key add /var/nvidia-diag-driver-local-repo-410.129/7fa2af80.pub
sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1804-410.129_1.0-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-drivers
sudo reboot

nvidia-smi -q | head

git clone https://github.com/LucasBamidele/TCC2.git
cd TCC2
./install_dep_aws.sh

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

tmux new -s mywindow

tmux a -t mywindow

# talvez tentar fazer isso primeiro
#sudo apt-get install nvidia-cuda-toolkit
# sudo reboot