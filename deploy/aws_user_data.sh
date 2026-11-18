#!/bin/bash
# runs once when the EC2 server first boots
exec > /var/log/app-setup.log 2>&1
set -ex

dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user

mkdir -p /usr/libexec/docker/cli-plugins
curl -fsSL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/libexec/docker/cli-plugins/docker-compose
chmod +x /usr/libexec/docker/cli-plugins/docker-compose

# 2 GB swap as a safety net while a retraining run is going
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

git clone https://github.com/kbelyse/pneumonia-xray-mlops.git /opt/app
cd /opt/app
DOCKER_BUILDKIT=0 docker build -t pneumonia-xray-api .
HTTP_PORT=80 docker compose up -d --no-build --scale api=2
echo SETUP_DONE
