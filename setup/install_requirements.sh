#!/bin/bash

# echo "Install cudatoolkit"
# conda install -y -c anaconda cudatoolkit

echo "Installing dependencies for pip"
poetry add torch==1.4.0
poetry add torchvision==0.5.0
poetry add matplotlib==2.2.5
poetry add tqdm==4.32.1
poetry add open3d==0.9.0
poetry add trimesh==3.7.12
poetry add scipy==1.5.1    

# echo "Installing local cuda emd dependency"
# cd metrics/PyTorchEMD
# python setup.py install
# cd ../..