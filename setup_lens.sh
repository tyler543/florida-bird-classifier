#!/bin/bash
set -e

echo "=== cef168 Canon Lens Driver Setup ==="

echo "[1/4] Cloning cef168..."
git clone https://github.com/pinefeat/cef168 /tmp/cef168
cd /tmp/cef168

echo "[2/4] Configuring for Camera Module 3 (IMX708)..."
bash configure.sh imx708

echo "[3/4] Building and installing kernel driver..."
make
sudo make install

echo "[4/4] Cleaning up..."
cd ~
rm -rf /tmp/cef168

echo ""
echo "=== Done! Rebooting in 5 seconds... ==="
echo "After reboot, SSH back in and run: sudo calibrate"
echo "Then follow the prompts to calibrate your specific Canon lens."
sleep 5
sudo reboot
