#!bin/bash

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

conda create -n beautifulmind python=3.11 && echo "conda activate beautifulmind" > ~/.bashrc

source ~/.bashrc

# Environment for module question_answering
#pip install -r src/question_answering/requirements.txt

# Environment for module chatbot

#python -m ipykernel install --user --name=beautifulmind