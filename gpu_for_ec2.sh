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

GRUB_CMDLINE_LINUX="rdblacklist=nouveau"

sudo update-grub

# Download the driver package that you identified earlier as follows.

# For P2 and P3 instances, the following command downloads the NVIDIA driver,
# where xxx.xxx represents the version of the NVIDIA driver.

wget http://us.download.nvidia.com/tesla/418.87/nvidia-driver-local-repo-ubuntu1804-418.87.00_1.0-1_amd64.deb
dpkg -i nvidia-driver-local-repo-ubuntu1804-418.87.00_1.0-1_amd64.deb
apt-get update
apt-get install cuda-drivers
sudo reboot

nvidia-smi -q | head