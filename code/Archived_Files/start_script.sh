#!/bin/bash
apt install unzip #install unzip
apt install wget #install wget

#download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

#launch miniconda
bash ~/miniconda.sh -b -p $HOME/miniconda

#set miniconda activation
source /root/miniconda/bin/activate
conda init

#create gpu env
conda create -n gpu python=3.7 -y
conda activate gpu

#install packages
conda install pandas numpy pytorch torchvision opencv ipykernel jupyter ipython matplotlib  -y
conda install pillow=6.2.1 -y

#set to jupyter environment
python -m ipykernel install --user --name=gpu 
git clone https://github.com/bwolfson2/dl2020.git #if there is no git, do "apt install git" first
cd dl2020

#download gcloud sdk
wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-290.0.0-linux-x86_64.tar.gz
tar -xvf google-cloud-sdk-290.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh --path-update true -q

#Open new terminal
cd dl2020 #or make sure you are in directory /workspace/dl2020
mv client.zip2 client.zip
unzip client.zip

#to upload files into the google cloud account
./bucket_upload.sh <file name>

#to download files from the google account
gsutil cp gs://dl2020/<file name> <file name>

#to look at the google account contents : https://console.cloud.google.com/storage/browser/dl2020


#download student data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fCYFNpLopbUDOc5Pv3Gv1VD6GCVb5ash' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fCYFNpLopbUDOc5Pv3Gv1VD6GCVb5ash" -O student_data.zip && rm -rf /tmp/cookies.txt
unzip student_data.zip
rm student_data.zip